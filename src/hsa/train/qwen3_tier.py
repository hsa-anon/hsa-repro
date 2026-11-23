import logging

import torch

from hsa.torch_tier_fast import three_tier_attention_fast, three_tier_attention_train

logger = logging.getLogger(__name__)


def quest_dense_prefill(q, k, v, q_offset):
    # Dense causal attention over the full history for the Quest baseline's
    # prefill path: q (B, Hq, t, D) post-RoPE, k/v (B, Hkv, T, D). GQA is
    # expanded by repeat_interleave; the mask allows key j for query i iff
    # j <= q_offset + i (True = attend in SDPA bool-mask semantics).
    import torch.nn.functional as F

    _B, Hq, t, _D = q.shape
    Hkv = k.shape[1]
    T = k.shape[2]
    group = Hq // Hkv
    ke = k.repeat_interleave(group, dim=1)
    ve = v.repeat_interleave(group, dim=1)
    j = torch.arange(T, device=q.device)
    i = torch.arange(q_offset, q_offset + t, device=q.device)
    mask = j.unsqueeze(0) <= i.unsqueeze(1)  # (t, T)
    return F.scaled_dot_product_attention(q, ke, ve, attn_mask=mask)


def full_attn_layers(model) -> list:
    # Qwen3.8-27B is hybrid: 64 layers, 48 GatedDeltaNet (linear attention)
    # + 16 full self_attn at indices congruent 3 mod 4. Tier injection only
    # makes sense on the full-attention layers.
    return [l for l in model.model.layers if hasattr(l, "self_attn")]


def attach_tier_weights(model, d_latent: int, std: float = 1e-3) -> None:
    # Per-layer latent projections on every full-attention module. Asymmetric
    # init: W_q_lat tiny random (gradient carrier), W_k_lat zeros. Dtype
    # follows the model (bf16 training keeps matmuls uniform).
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    layers = full_attn_layers(model)
    for layer in layers:
        attn = layer.self_attn
        hq = attn.config.num_attention_heads
        hkv = attn.config.num_key_value_heads
        d_head = attn.head_dim
        attn.w_q_lat = torch.nn.Parameter(
            (torch.randn(hq * d_head, d_latent, device=device) * std).to(dtype)
        )
        attn.w_k_lat = torch.nn.Parameter(
            torch.zeros(hkv * d_head, d_latent, dtype=dtype, device=device)
        )
        attn.tier_window = None
        attn.tier_block = None
    logger.info("attached tier weights d_latent=%d to %d full-attn layers", d_latent, len(layers))


def make_tier_forward(orig_forward):
    import transformers.models.qwen3.modeling_qwen3 as m3
    import transformers.models.qwen3_5.modeling_qwen3_5 as m35

    def tier_forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        # Qwen3.5 output gating: q_proj emits 2x head dim, second half is a
        # sigmoid gate multiplied onto the attention output before o_proj.
        # Plain Qwen3 has no gate; detect via q_proj out_features.
        gated = self.q_proj.out_features > self.config.num_attention_heads * self.head_dim
        m = m35 if gated else m3
        input_shape = hidden_states.shape[:-1]
        hidden_shape_q = (*input_shape, -1, self.head_dim)
        gate = None
        if gated:
            q_gates = self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2)
            query_states, gate = torch.chunk(q_gates, 2, dim=-1)
            gate = gate.reshape(*input_shape, -1)
        else:
            query_states = self.q_proj(hidden_states).view(hidden_shape_q)
        query_states = self.q_norm(query_states).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape_q)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape_q).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = m.apply_rotary_pos_emb(query_states, key_states, cos, sin)
        b, h, t, d = query_states.shape
        if (
            getattr(self, "_tier_decode", False)
            and t == 1
            and getattr(self, "_tier_cache", None) is not None
        ):
            # Decode step: T=1 with history in our own TierLayerCache (HF
            # cache stays empty for tier layers; the GatedDeltaNet layers
            # own the official cache so position ids stay correct).
            from hsa.eval.nsa_cache import NsaLayerCache
            from hsa.eval.quest_cache import QuestLayerCache
            from hsa.eval.tier_cache import TierLayerCache

            if isinstance(self._tier_cache, NsaLayerCache):
                # NSA external baseline: three-branch selection decode.
                out1 = self._tier_cache.decode(
                    query_states[0, :, 0], key_states[0, :, 0], value_states[0, :, 0]
                )
            elif isinstance(self._tier_cache, QuestLayerCache):
                # Quest external baseline: page-bound selection, no tiers.
                out1 = self._tier_cache.decode(
                    query_states[0, :, 0],
                    key_states[0, :, 0],
                    value_states[0, :, 0],
                    pages=self._decode_quest,
                )
            else:
                out1 = self._tier_cache.decode(
                    query_states[0, :, 0],
                    key_states[0, :, 0],
                    value_states[0, :, 0],
                    self.w_q_lat,
                    self.w_k_lat,
                    latent_clamp=getattr(self, "_latent_clamp", None),
                    topk=getattr(self, "_decode_topk", None),
                )
            attn_output = out1.view(1, 1, h * d)
            if gated:
                attn_output = attn_output * torch.sigmoid(gate)
            return self.o_proj(attn_output), None
        q_offset = 0
        quest_pages = getattr(self, "_decode_quest", None)
        nsa_blocks = getattr(self, "_decode_nsa", None)
        if getattr(self, "_tier_decode", False) and t > 1:
            # Chunked prefill: this chunk's queries sit at absolute positions
            # [history_len, history_len + t) and must attend over the FULL
            # history, not just the chunk - otherwise chunk 2+ loses all
            # long-range context during prefill (the 32k sweep failure).
            from hsa.eval.nsa_cache import NsaLayerCache
            from hsa.eval.quest_cache import QuestLayerCache
            from hsa.eval.tier_cache import TierLayerCache

            if nsa_blocks:
                # NSA baseline: dense KV storage for decode-time selection;
                # prefill attention itself stays dense (matched to Quest).
                if getattr(self, "_tier_cache", None) is None:
                    self._tier_cache = NsaLayerCache(
                        block_size=32, n_blocks=nsa_blocks, window=self.tier_window
                    )
                    self._tier_cache.seed(
                        key_states[0].transpose(0, 1), value_states[0].transpose(0, 1)
                    )
                else:
                    q_offset = int(self._tier_cache.total)
                    self._tier_cache.append_chunk(
                        key_states[0].transpose(0, 1), value_states[0].transpose(0, 1)
                    )
            elif quest_pages:
                # Quest baseline: dense KV storage for decode-time page
                # selection; prefill attention itself stays dense.
                if getattr(self, "_tier_cache", None) is None:
                    self._tier_cache = QuestLayerCache(page_size=64)
                    self._tier_cache.seed(
                        key_states[0].transpose(0, 1), value_states[0].transpose(0, 1)
                    )
                else:
                    q_offset = int(self._tier_cache.total)
                    self._tier_cache.append_chunk(
                        key_states[0].transpose(0, 1), value_states[0].transpose(0, 1)
                    )
            else:
                if getattr(self, "_tier_cache", None) is None:
                    self._tier_cache = TierLayerCache(self.tier_window, self.tier_block)
                    self._tier_cache.seed(
                        key_states[0].transpose(0, 1), value_states[0].transpose(0, 1)
                    )
                else:
                    q_offset = int(self._tier_cache.total)
                    self._tier_cache.seed(
                        torch.cat([self._tier_cache.k, key_states[0].transpose(0, 1)], dim=0),
                        torch.cat([self._tier_cache.v, value_states[0].transpose(0, 1)], dim=0),
                    )
            k_hist = self._tier_cache.k.unsqueeze(0)
            v_hist = self._tier_cache.v.unsqueeze(0)
        else:
            # Kernel contract is (B, T, Hkv, Dv); key/value states come out of
            # the attention module in HF layout (B, Hkv, T, D).
            k_hist = key_states.transpose(1, 2)
            v_hist = value_states.transpose(1, 2)
        if quest_pages or nsa_blocks:
            # External-baseline prefill (Quest / NSA): dense SDPA over the
            # full history.
            out = quest_dense_prefill(
                query_states, k_hist.transpose(1, 2), v_hist.transpose(1, 2), q_offset
            )
        elif torch.is_grad_enabled() and q_offset == 0:
            # Training path: custom Function, forward fully no_grad, analytic
            # latent-only backward (see _TierTrainFn) - keeps the 32k stage
            # under the memory wall by never materializing the graph-mode
            # full-sequence fp32 softmax.
            out, _ = three_tier_attention_train(
                query_states.transpose(1, 2),
                k_hist,
                v_hist,
                self.tier_window,
                self.tier_block,
                self.w_q_lat,
                self.w_k_lat,
                block_target=getattr(self, "_block_target", None),
                block_weight=getattr(self, "_block_weight", 1.0),
            )
        else:
            out, _ = three_tier_attention_fast(
                query_states.transpose(1, 2),
                k_hist,
                v_hist,
                self.tier_window,
                self.tier_block,
                w_q_lat=self.w_q_lat,
                w_k_lat=self.w_k_lat,
                latent_clamp=getattr(self, "_latent_clamp", None),
                latent_zscore=getattr(self, "_latent_zscore", False),
                q_offset=q_offset,
            )
        attn_output = out.transpose(1, 2).reshape(b, t, h * d)
        if gated:
            attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output), None

    return tier_forward


def enable_tier_mode(model, window: int, block_size: int) -> int:
    n = 0
    for layer in full_attn_layers(model):
        attn = layer.self_attn
        if not hasattr(attn, "w_q_lat"):
            raise RuntimeError("attach_tier_weights must run before enable_tier_mode")
        if not getattr(attn, "_orig_forward", None):
            attn._orig_forward = attn.forward
        attn.forward = make_tier_forward(attn._orig_forward).__get__(attn)
        attn.tier_window = window
        attn.tier_block = block_size
        n += 1
    logger.info(
        "tier mode enabled on %d full-attn layers window=%d block=%d", n, window, block_size
    )
    return n


def disable_tier_mode(model) -> None:
    for layer in full_attn_layers(model):
        attn = layer.self_attn
        if getattr(attn, "_orig_forward", None):
            attn.forward = attn._orig_forward
            attn._orig_forward = None
    logger.info("tier mode disabled (dense teacher path restored)")


def trainable_parameters(model) -> list:
    return [
        p
        for layer in full_attn_layers(model)
        for p in (layer.self_attn.w_q_lat, layer.self_attn.w_k_lat)
    ]
