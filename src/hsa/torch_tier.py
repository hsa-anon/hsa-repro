import logging

import torch

logger = logging.getLogger(__name__)


def _default_latent_project(x: torch.Tensor, d_latent: int) -> torch.Tensor:
    d = x.shape[-1]
    if d_latent <= d:
        return x[..., :d_latent]
    pad = torch.zeros(*x.shape[:-1], d_latent - d, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=-1)


def three_tier_attention(q, k, v, window: int, block_size: int, d_latent: int, w_k=None, w_v=None):
    """Combined three-tier attention in torch (training-side reference).

    Per query: exact window + exact landmark keys (block first tokens) +
    projected block-mean latents, one softmax over concatenated logits.
    q: (seq, h, d); k/v: (seq, h, d|dv). Returns (out (seq, h, dv), aux dict
    with per-block k_lat/v_lat for the training loss).
    """
    seq, h, d = q.shape
    dv = v.shape[-1]
    n_blocks = (seq + block_size - 1) // block_size
    counts = torch.tensor(
        [min((b + 1) * block_size, seq) - b * block_size for b in range(n_blocks)],
        dtype=q.dtype,
        device=q.device,
    )[:, None, None]
    pad_t = n_blocks * block_size - seq
    if pad_t:
        k = torch.cat([k, torch.zeros(pad_t, h, d, dtype=q.dtype, device=q.device)], dim=0)
        v = torch.cat([v, torch.zeros(pad_t, h, dv, dtype=v.dtype, device=v.device)], dim=0)
    k_mean = k.reshape(n_blocks, block_size, h, d).sum(dim=1) / counts
    v_mean = v.reshape(n_blocks, block_size, h, dv).sum(dim=1) / counts
    if w_k is not None:
        k_lat = torch.einsum("bhd,de->bhe", k_mean, w_k)
    else:
        k_lat = _default_latent_project(k_mean, d_latent)
    v_lat = torch.einsum("bhd,de->bhe", v_mean, w_v) if w_v is not None else v_mean
    s_d = d**0.5
    s_l = d_latent**0.5
    out = torch.zeros(seq, h, dv, dtype=q.dtype, device=q.device)
    max_vis = 0
    for i in range(seq):
        ws = max(0, i - window + 1)
        wpos = list(range(ws, i + 1))
        visible = [b for b in range(n_blocks) if (b + 1) * block_size - 1 < ws]
        landmarks = [b * block_size for b in visible]
        sw = torch.einsum("hd,khd->hk", q[i], k[wpos]) / s_d
        parts = [sw]
        if visible:
            sl_ = torch.einsum("hd,khd->hk", q[i], k[landmarks]) / s_d
            ql = (
                torch.einsum("hd,de->he", q[i], w_k)
                if w_k is not None
                else _default_latent_project(q[i], d_latent)
            )
            slt = torch.einsum("he,bhe->hb", ql, k_lat[visible]) / s_l
            parts += [sl_, slt]
        logits = torch.cat(parts, dim=-1)
        p = torch.softmax(logits, dim=-1)
        o = torch.einsum("hk,khd->hd", p[:, : len(wpos)], v[wpos])
        if visible:
            n_w = len(wpos)
            o = o + torch.einsum("hk,khd->hd", p[:, n_w : n_w + len(landmarks)], v[landmarks])
            o = o + torch.einsum("hb,bhd->hd", p[:, n_w + len(landmarks) :], v_lat[visible])
            max_vis = max(max_vis, len(visible))
        out[i] = o
    aux = {"k_lat": k_lat, "v_lat": v_lat, "max_visible_blocks": max_vis}
    logger.debug("three-tier torch pass seq=%d blocks=%d", seq, n_blocks)
    return out, aux
