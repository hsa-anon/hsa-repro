import logging

import mlx.core as mx

logger = logging.getLogger(__name__)


def project_to_latent(x: mx.array, d_latent: int) -> mx.array:
    """Deterministic truncation/pad projection to the latent bottleneck space.
    Learned projections replace this in M2; the oracle pins the contract."""
    d = x.shape[-1]
    if d_latent <= d:
        return x[..., :d_latent]
    pad = mx.zeros((*x.shape[:-1], d_latent - d), dtype=x.dtype)
    return mx.concatenate([x, pad], axis=-1)


def block_latents(
    k: mx.array, v: mx.array, block_size: int, d_latent: int
) -> tuple[mx.array, mx.array]:
    """Per-block latent keys (projected block-mean) and latent values (block-mean).

    k: (seq, h, d), v: (seq, h, d_v) -> latents (n_blocks, h, d_latent), (n_blocks, h, d_v)
    """
    seq = k.shape[0]
    n_blocks = (seq + block_size - 1) // block_size
    counts = mx.array(
        [min((b + 1) * block_size, seq) - b * block_size for b in range(n_blocks)],
        dtype=k.dtype,
    )[:, None, None]
    pad = n_blocks * block_size - seq
    if pad:
        k = mx.concatenate([k, mx.zeros((pad, *k.shape[1:]), dtype=k.dtype)], axis=0)
        v = mx.concatenate([v, mx.zeros((pad, *v.shape[1:]), dtype=v.dtype)], axis=0)
    kb = mx.transpose(k.reshape(n_blocks, block_size, *k.shape[1:]), (0, 2, 1, 3))
    vb = mx.transpose(v.reshape(n_blocks, block_size, *v.shape[1:]), (0, 2, 1, 3))
    k_mean = kb.sum(axis=2) / counts
    v_mean = vb.sum(axis=2) / counts
    return project_to_latent(k_mean, d_latent), v_mean


def latent_attention(q, k, v, window: int, block_size: int, d_latent: int) -> mx.array:
    """Latent tier of the three-tier scheme.

    Per query: exact window attention (d-space scores) + per-block latent
    attention for completed blocks fully below the window (latent-space
    scores). Single softmax over concatenated logits.

    q: (seq, hq, d); k/v: (seq, hkv, d). GQA repeated. Returns (seq, hq, d).
    """
    seq, hq, d = q.shape
    hkv = k.shape[1]
    group = hq // hkv
    if group > 1:
        k = mx.repeat(k, group, axis=1)
        v = mx.repeat(v, group, axis=1)
    k_lat, v_lat = block_latents(k, v, block_size, d_latent)
    n_blocks = k_lat.shape[0]
    q_lat = project_to_latent(q, d_latent)

    idx = mx.arange(seq)
    dist = idx[:, None] - idx[None, :]
    in_window = (dist >= 0) & (dist < window)

    qw = mx.transpose(q, (1, 0, 2))
    kh = mx.transpose(k, (1, 2, 0))
    vh = mx.transpose(v, (1, 0, 2))
    sw = (qw @ kh) / mx.sqrt(mx.array(float(d), mx.float32))
    sw = mx.where(in_window[None], sw, mx.array(-mx.inf, sw.dtype))

    qh = mx.transpose(q_lat, (1, 0, 2))
    klh = mx.transpose(k_lat, (1, 2, 0))
    sl = (qh @ klh) / mx.sqrt(mx.array(float(d_latent), mx.float32))
    # block b covers positions [b*bs, (b+1)*bs); visible iff its last position
    # is below the query's window start
    block_last = mx.array([min((b + 1) * block_size, seq) - 1 for b in range(n_blocks)])
    window_start = mx.maximum(idx - window + 1, 0)
    lat_visible = block_last[None, :] < window_start[:, None]
    sl = mx.where(lat_visible[None], sl, mx.array(-mx.inf, sl.dtype))

    scores = mx.concatenate([sw, sl], axis=-1)
    probs = mx.softmax(scores, axis=-1)
    pw, pl = probs[..., :seq], probs[..., seq:]
    out = pw @ vh + pl @ mx.transpose(v_lat, (1, 0, 2))
    out = mx.transpose(out, (1, 0, 2))
    logger.debug(
        "latent tier seq=%d window=%d block=%d d_lat=%d", seq, window, block_size, d_latent
    )
    return out
