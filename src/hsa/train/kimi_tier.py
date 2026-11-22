import logging

import torch

logger = logging.getLogger(__name__)


class KimiTierCache:
    # Per-layer compressed MLA cache for tiered decode: stores the raw
    # kv_a projection per token ([kv_lora_rank, qk_rope_head_dim] = 576
    # floats on the 48B model) instead of the expanded per-head K/V the
    # official KimiDynamicCache keeps (H * (q_head_dim + v_head_dim) per
    # token, the only O(T) memory component of the hybrid). Expansion to
    # per-head K/V happens on demand through kv_b_proj, restricted to the
    # retrieved/window/sink tokens at decode.

    def __init__(self, attn):
        self.attn = attn
        self.compressed: torch.Tensor | None = None  # [T, kv_lora + qk_rope]
        self.total = 0

    def append(self, compressed: torch.Tensor) -> int:
        # compressed [B, T, kv_lora + qk_rope]; batch 1 (eval path).
        chunk = compressed[0].detach()
        offset = self.total
        if self.compressed is None:
            self.compressed = chunk
        else:
            self.compressed = torch.cat([self.compressed, chunk], dim=0)
        self.total = self.compressed.shape[0]
        return offset

    def expand(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # rows [n] positions -> per-head k [n, H, q_head_dim], v [n, H, v_head].
        # Mirrors the official forward: layernorm the compressed part, expand
        # through kv_b_proj, split nope/v, broadcast the rope channel.
        attn = self.attn
        c_sel = self.compressed[rows]
        d_kv = attn.kv_lora_rank
        c_norm = attn.kv_a_layernorm(c_sel[..., :d_kv])
        krot = c_sel[..., d_kv:]
        kv = attn.kv_b_proj(c_norm).view(
            -1, attn.num_heads, attn.qk_nope_head_dim + attn.v_head_dim
        )
        k_nope, v = torch.split(kv, [attn.qk_nope_head_dim, attn.v_head_dim], dim=-1)
        k = torch.cat([k_nope, krot[:, None, :].expand(-1, attn.num_heads, -1)], dim=-1)
        return k, v

    def decode(
        self,
        query: torch.Tensor,
        w_q_lat: torch.nn.Parameter,
        w_k_lat: torch.nn.Parameter,
        topk: int | None = None,
        window: int = 0,
        sink: int = 0,
    ) -> torch.Tensor:
        # query [H, q_head_dim] at the just-appended position; selection is
        # query-dependent (Pivot A): latent index scores over the compressed
        # cache, then top-k + last window + first sink positions are expanded.
        # topk=None disables the cap and scores every position (exact dense
        # control used for parity tests).
        attn = self.attn
        total = self.total
        device = query.device
        zq = query.reshape(-1).float() @ w_q_lat.float()
        zk = self.compressed.float() @ w_k_lat.float()
        scores = zk @ zq
        candidates = torch.arange(total, device=device)
        if topk is not None and topk < total:
            k = min(topk, max(total - window - sink, 0))
            top = torch.topk(scores[:-window] if window else scores, k).indices
            picked = torch.cat([candidates[:sink], top, candidates[total - window :]])
        else:
            picked = candidates
        picked = torch.unique(picked, sorted=True)
        k_exp, v_exp = self.expand(picked)  # [n, H, 192], [n, H, 128]
        scores_attn = torch.einsum("hd,nhd->hn", query, k_exp) * attn.q_head_dim**-0.5
        out = torch.einsum("hn,nhv->hv", torch.softmax(scores_attn, dim=-1), v_exp)
        return out.reshape(1, 1, attn.num_heads * attn.v_head_dim)


def mla_layers(model) -> list:
    # Kimi Linear is hybrid: config.linear_attn_config.full_attn_layers lists
    # the global MLA layers (3:1 KDA-to-MLA). The shipped config lists index
    # 27 for a 27-layer model (0..26), so out-of-range entries are dropped
    # with a warning instead of crashing at load.
    cfg = model.config
    linear_cfg = getattr(cfg, "linear_attn_config", None)
    if linear_cfg is None or not hasattr(linear_cfg, "full_attn_layers"):
        raise ValueError("model config carries no linear_attn_config.full_attn_layers")
    layers = model.model.layers
    depth = cfg.num_hidden_layers
    picked = []
    for i in linear_cfg.full_attn_layers:
        if i >= depth:
            logger.warning("dropping out-of-range full-attn layer index %d (depth %d)", i, depth)
        elif hasattr(layers[i], "self_attn"):
            picked.append(layers[i])
        else:
            logger.warning("layer %d has no self_attn (KDA layer), skipping", i)
    if not picked:
        raise ValueError("no MLA layers resolved from full_attn_layers")
    return picked


def attach_tier_weights(model, d_latent: int, std: float = 1e-3) -> None:
    # Latent index projections on every MLA module. The key side indexes the
    # raw per-token compressed vector (kv_lora_rank + qk_rope_head_dim); the
    # query side indexes the full per-head q (nope + rope channels, no q_lora
    # and no q_norm in Kimi MLA). Asymmetric init mirrors qwen3_tier.
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    layers = mla_layers(model)
    for layer in layers:
        attn = layer.self_attn
        in_kv = attn.kv_lora_rank + attn.qk_rope_head_dim
        attn.w_q_lat = torch.nn.Parameter(
            (torch.randn(attn.num_heads * attn.q_head_dim, d_latent, device=device) * std).to(dtype)
        )
        attn.w_k_lat = torch.nn.Parameter(torch.zeros(in_kv, d_latent, dtype=dtype, device=device))
    logger.info("attached tier weights d_latent=%d to %d MLA layers", d_latent, len(layers))


def _dense_forward(attn, hidden_states, k_hist, v_hist, q_offset, n_chunk):
    # Dense SDPA over the full history with causal mask; k/v are expanded
    # [T, H, q_head_dim] / [T, H, v_head]. Used for chunked prefill over the
    # compressed history: each chunk re-expands the whole cache, which keeps
    # prefill exact while storage stays at the 576-float-per-token footprint.
    import torch.nn.functional as F

    b = hidden_states.shape[0]
    t = n_chunk
    h = k_hist.shape[1]
    q = attn.q_proj(hidden_states).view(b, t, h, -1).transpose(1, 2)
    total = k_hist.shape[0]
    j = torch.arange(total, device=hidden_states.device)
    i = torch.arange(q_offset, q_offset + t, device=hidden_states.device)
    mask = j.unsqueeze(0) <= i.unsqueeze(1)
    out = F.scaled_dot_product_attention(
        q, k_hist.transpose(0, 1).unsqueeze(0), v_hist.transpose(0, 1).unsqueeze(0), attn_mask=mask
    )
    return out.transpose(1, 2).reshape(b, t, h * v_hist.shape[2])


def make_tier_forward(orig_forward):
    def tier_forward(
        self,
        hidden_states,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        # MLA tier forward. The official module applies no RoPE (the rope
        # channels are plain learned projections) and no q_norm, so the
        # computation is a straight linear decomposition; the compressed
        # cache replaces the expanded K/V cache past_key_values would hold.
        compressed = self.kv_a_proj_with_mqa(hidden_states)
        seq_len = hidden_states.shape[1]
        decode = (
            getattr(self, "_tier_decode", False) and getattr(self, "_tier_cache", None) is not None
        )
        if decode and seq_len == 1:
            cache = self._tier_cache
            cache.append(compressed)
            query = self.q_proj(hidden_states)[0, 0].view(self.num_heads, self.q_head_dim)
            out = cache.decode(
                query,
                self.w_q_lat,
                self.w_k_lat,
                topk=getattr(self, "_decode_topk", None),
                window=getattr(self, "_decode_window", 0),
                sink=getattr(self, "_decode_sink", 0),
            )
            return self.o_proj(out.to(hidden_states.dtype)), None
        if decode:
            cache = self._tier_cache
            q_offset = cache.append(compressed)
            k_all, v_all = cache.expand(torch.arange(cache.total, device=compressed.device))
            out = _dense_forward(self, hidden_states, k_all, v_all, q_offset, seq_len)
            return self.o_proj(out.to(hidden_states.dtype)), None
        # Teacher/parity path: the official expanded forward, untouched.
        return orig_forward(self, hidden_states, attention_mask, past_key_values, **kwargs)

    return tier_forward


def enable_tier_mode(model, window: int = 512, sink: int = 0, topk: int | None = None) -> int:
    # Swap every MLA forward for the tiered version. Decode-time retrieval
    # knobs live on the module: _decode_topk / _decode_window / _decode_sink.
    # topk=None keeps decode exact-dense (the Pivot A step-2 control); the
    # Ward-bucket index (hsa.index) plugs in as the step-3 selector later.
    n = 0
    for layer in mla_layers(model):
        attn = layer.self_attn
        if not hasattr(attn, "w_q_lat"):
            raise RuntimeError("attach_tier_weights must run before enable_tier_mode")
        if not getattr(attn, "_orig_forward", None):
            attn._orig_forward = attn.forward
        attn.forward = make_tier_forward(attn._orig_forward).__get__(attn)
        attn._tier_cache = KimiTierCache(attn)
        attn._decode_window = window
        attn._decode_sink = sink
        attn._decode_topk = topk
        n += 1
    logger.info(
        "tier mode enabled on %d MLA layers window=%d sink=%d topk=%s", n, window, sink, topk
    )
    return n


def disable_tier_mode(model) -> None:
    for layer in mla_layers(model):
        attn = layer.self_attn
        if getattr(attn, "_orig_forward", None):
            attn.forward = attn._orig_forward
            attn._orig_forward = None
        attn._tier_cache = None
    logger.info("tier mode disabled (dense MLA path restored)")


def trainable_parameters(model) -> list:
    return [
        p for layer in mla_layers(model) for p in (layer.self_attn.w_q_lat, layer.self_attn.w_k_lat)
    ]
