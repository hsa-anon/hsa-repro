import logging

import torch

logger = logging.getLogger(__name__)


def capture_teacher_attn_inputs(model):
    # Pre-forward hooks on every full-attention module during the dense
    # teacher pass: capture each layer's input hidden states + position
    # embeddings so per-block attention mass can be derived afterwards.
    captures = {}
    handles = []

    def mk(idx):
        def pre(module, args, kwargs):
            hs = args[0] if args else kwargs.get("hidden_states")
            pe = kwargs.get("position_embeddings")
            if pe is None and len(args) > 1:
                pe = args[1]
            captures[idx] = (hs.detach(), pe)

        return pre

    for i, layer in enumerate([l for l in model.model.layers if hasattr(l, "self_attn")]):
        handles.append(layer.self_attn.register_forward_pre_hook(mk(i), with_kwargs=True))
    logger.info("block-mass capture hooks on %d full-attn layers", len(handles))
    return captures, handles


def release_capture_hooks(handles):
    for h in handles:
        h.remove()


def compute_block_mass_targets(model, captures, rows, window, block_size, device):
    # Per full-attention layer: recompute teacher q/k exactly as the dense
    # path does (qwen3_tier.tier_forward lines 44-59 recipe), take the
    # attention rows for the sampled query positions, softmax over the
    # causal row, sum per landmark block, head-average, renormalize over
    # blocks visible to the student latent tier. Returns
    # {attn_module: (target (K, n_blocks) float32, mean_entropy)}.
    import transformers.models.qwen3.modeling_qwen3 as m3
    import transformers.models.qwen3_5.modeling_qwen3_5 as m35

    ridx = torch.as_tensor(rows, device=device, dtype=torch.long)
    ridx_dev = ridx.clone()
    targets = {}
    with torch.no_grad():
        for i, layer in enumerate([l for l in model.model.layers if hasattr(l, "self_attn")]):
            hs, pe = captures.pop(i)
            hs = hs if hs.device == device else hs.to(device)
            attn = layer.self_attn
            head_dim = attn.head_dim
            hq = attn.config.num_attention_heads
            hkv = attn.config.num_key_value_heads
            T = hs.shape[1]
            nb = (T + block_size - 1) // block_size
            gated = attn.q_proj.out_features > hq * head_dim
            m = m35 if gated else m3
            q_in = hs[:, ridx]
            if gated:
                q_gates = attn.q_proj(q_in).view(-1, hq, head_dim * 2)
                q, _ = torch.chunk(q_gates, 2, dim=-1)
            else:
                q = attn.q_proj(q_in).view(-1, hq, head_dim)
            q = attn.q_norm(q).transpose(0, 1)
            k = attn.k_norm(attn.k_proj(hs).view(T, hkv, head_dim)).transpose(0, 1)
            cos, sin = pe
            qr, _ = m.apply_rotary_pos_emb(
                q.unsqueeze(0), q.unsqueeze(0), cos[:, ridx], sin[:, ridx]
            )
            kr, _ = m.apply_rotary_pos_emb(k.unsqueeze(0), k.unsqueeze(0), cos, sin)
            # GQA: expand the kv heads to the query head count (group 6 on
            # Qwen3.8-27B) so the score rows line up per query head.
            kr_full = kr[0].repeat_interleave(hq // hkv, dim=0)
            scores = torch.einsum("hkd,htd->hkt", qr[0].float(), kr_full.float()) / head_dim**0.5
            causal = ridx[:, None] >= torch.arange(T, device=device)[None, :]
            scores = scores.masked_fill(~causal[None], torch.finfo(torch.float32).min)
            probs = torch.softmax(scores, dim=-1)  # (Hq, K, T)
            pad = nb * block_size - T
            if pad:
                probs = torch.nn.functional.pad(probs, (0, pad))
            block_mass = probs.view(hq, -1, nb, block_size).sum(-1).mean(0)  # (K, nb)
            block_last = torch.tensor(
                [min((b + 1) * block_size, T) - 1 for b in range(nb)], device=device
            )
            ws = (ridx - window + 1).clamp(min=0)
            vis = block_last[None, :] < ws[:, None]  # (K, nb)
            mass_vis = block_mass * vis
            denom = mass_vis.sum(-1, keepdim=True).clamp_min(1e-8)
            target = (mass_vis / denom).float()
            ent = -(target * target.clamp_min(1e-9).log()).sum(-1).mean()
            targets[attn] = (ridx_dev, target, float(ent))
            del hs, q, k, qr, kr, scores, probs
    ent_mean = sum(e for _, _, e in targets.values()) / max(len(targets), 1)
    logger.info(
        "block-mass targets: %d layers rows=%d mean entropy %.4f", len(targets), len(ridx), ent_mean
    )
    return targets
