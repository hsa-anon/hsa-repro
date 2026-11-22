import logging

import mlx.core as mx

logger = logging.getLogger(__name__)


def three_tier_attention_fast_mlx(
    q, k, v, window, block_size, w_q_lat=None, w_k_lat=None, latent_clamp=None, q_offset=0
):
    # MLX port of hsa.torch_tier_fast.three_tier_attention_fast, single-request
    # serving form (B=1). Shapes: q (Tq, Hq, D), k/v (Tabs, Hkv, Dv). The query
    # rows are the absolute positions [q_offset, q_offset + Tq) - full-history
    # chunked prefill passes the last chunk as q with q_offset=chunk start and
    # the whole history as k/v; q_offset=0 with Tq==Tabs is the plain case.
    # Exact window + block-first-token landmarks + shared-latent gist: block-mean
    # k over concatenated kv heads -> w_k_lat (Hkv*Dv, L), token-concat query ->
    # w_q_lat (Hq*D, L), one latent logit per (query, block) broadcast across
    # heads; latent values = block-mean v per kv head, GQA-expanded. One
    # softmax per head over [window | landmarks | latent]. Returns (Tq, Hq, Dv).
    Tq, Hq, D = q.shape
    Hkv = k.shape[1]
    Dv = v.shape[-1]
    Tabs = k.shape[0]
    group = Hq // Hkv
    L = w_k_lat.shape[-1] if w_k_lat is not None else 0
    s_d = mx.sqrt(mx.array(float(D), mx.float32))
    s_l = mx.sqrt(mx.array(float(L), mx.float32)) if L else mx.array(1.0, mx.float32)

    q_pos = q_offset + mx.arange(Tq)
    n_blocks = (Tabs + block_size - 1) // block_size
    pad = n_blocks * block_size - Tabs
    if pad:
        k = mx.concatenate([k, mx.zeros((pad, Hkv, Dv))], axis=0)
        v = mx.concatenate([v, mx.zeros((pad, Hkv, Dv))], axis=0)

    k5 = k.reshape(n_blocks, block_size, Hkv, Dv)
    v5 = v.reshape(n_blocks, block_size, Hkv, Dv)
    counts = mx.array(
        [float(min((b + 1) * block_size, Tabs) - b * block_size) for b in range(n_blocks)]
    )[:, None, None]
    k_mean = k5.sum(axis=1) / counts
    v_mean = v5.sum(axis=1) / counts
    block_last = mx.array([min((b + 1) * block_size, Tabs) - 1 for b in range(n_blocks)])

    ws = mx.maximum(q_pos - window + 1, 0)
    lat_visible = block_last[None, :] < ws[:, None]  # (Tq, n_blocks)
    neg = -mx.inf

    q_g = q.reshape(Tq, Hkv, group, D)
    pos = mx.arange(window)
    w_off = q_pos[:, None] - window + 1 + pos[None, :]
    w_valid = (w_off >= ws[:, None]) & (w_off <= q_pos[:, None])  # (Tq, W)
    w_off = mx.maximum(w_off, 0)
    kw = mx.take(k, w_off, axis=0)  # (Tq, W, Hkv, Dv)
    kw = mx.transpose(kw, (0, 2, 3, 1))  # (Tq, Hkv, Dv, W)
    sw = mx.einsum("thgd,thdw->thgw", q_g, kw) / s_d  # (Tq, Hkv, group, W)
    sw = sw.reshape(Tq, Hq, window)
    sw = mx.where(w_valid[:, None, :], sw, neg)
    landmark_pos = mx.arange(n_blocks) * block_size
    lm_k = k[landmark_pos]  # (n_blocks, Hkv, Dv)
    lm_v = v[landmark_pos]
    lm_kp = mx.transpose(lm_k, (1, 2, 0))  # (Hkv, Dv, n_blocks)
    sl = mx.einsum("thgd,hdw->thgw", q_g, lm_kp) / s_d  # (Tq, Hkv, group, n_blocks)
    sl = sl.reshape(Tq, Hq, n_blocks)
    sl = mx.where(lat_visible[:, None, :], sl, neg)

    logits_exact = mx.concatenate([sw, sl], axis=-1)  # (Tq, Hq, W + n_blocks)
    if L:
        q_flat = q.reshape(Tq, Hq * D)
        k_flat_mean = k_mean.reshape(n_blocks, Hkv * Dv)
        # Latent logits in fp32: untrained latents sit at ~660 scale where
        # bf16/fp16 ULP is ~4, so sub-4 relative differences (the ranking
        # signal) round away - mirrors the torch kernel fix.
        q_lat = (q_flat @ w_q_lat).astype(mx.float32)  # (Tq, L)
        k_lat = (k_flat_mean @ w_k_lat).astype(mx.float32)  # (n_blocks, L)
        slt = mx.einsum("tl,bl->tb", q_lat, k_lat) / s_l  # (Tq, n_blocks)
        slt = mx.where(lat_visible, slt, neg)
        if latent_clamp is not None:
            slt = mx.clip(slt, -latent_clamp, latent_clamp)
        slt_b = mx.repeat(slt[:, None, :], Hq, axis=1)  # (Tq, Hq, n_blocks)
        logits = mx.concatenate([logits_exact, slt_b], axis=-1).astype(mx.float32)
    else:
        logits = logits_exact.astype(mx.float32)

    probs = mx.softmax(logits, axis=-1).astype(logits_exact.dtype)
    pw = probs[..., :window].reshape(Tq, Hkv, group, window)
    plm = probs[..., window : window + n_blocks].reshape(Tq, Hkv, group, n_blocks)
    vw = mx.take(v, w_off, axis=0)  # (Tq, W, Hkv, Dv)
    vw = mx.transpose(vw, (0, 2, 1, 3))  # (Tq, Hkv, W, Dv)
    out = mx.einsum("thgw,thwd->thgd", pw, vw)  # (Tq, Hkv, group, Dv)
    lm_vp = mx.transpose(lm_v, (1, 0, 2))  # (Hkv, n_blocks, Dv)
    out = out + mx.einsum("thgb,hbd->thgd", plm, lm_vp)
    if L:
        plt = probs[..., window + n_blocks :].reshape(Tq, Hkv, group, n_blocks)
        vm = mx.transpose(v_mean, (1, 0, 2))  # (Hkv, n_blocks, Dv)
        out = out + mx.einsum("thgb,hbd->thgd", plt, vm)
    out = out.reshape(Tq, Hq, Dv)
    logger.debug(
        "three-tier mlx Tq=%d offset=%d Tabs=%d Hq=%d window=%d blocks=%d latent=%d",
        Tq,
        q_offset,
        Tabs,
        Hq,
        window,
        n_blocks,
        L,
    )
    return out
