import logging

import mlx.core as mx

logger = logging.getLogger(__name__)


def sliding_window_attention(q: mx.array, k: mx.array, v: mx.array, window: int) -> mx.array:
    """Causal sliding-window attention.

    q: (seq, hq, d), k/v: (seq, hkv, d), GQA via repeat. fp32 compute.
    """
    seq, hq, d = q.shape
    hkv = k.shape[1]
    group = hq // hkv
    if group > 1:
        k = mx.repeat(k, group, axis=1)
        v = mx.repeat(v, group, axis=1)
    qh = mx.transpose(q, (1, 0, 2))
    kh = mx.transpose(k, (1, 2, 0))
    scores = (qh @ kh) / mx.sqrt(mx.array(float(d), mx.float32))
    idx = mx.arange(seq)
    dist = idx[:, None] - idx[None, :]
    mask = (dist >= 0) & (dist < window)
    scores = mx.where(mask[None], scores, mx.array(-mx.inf, scores.dtype))
    probs = mx.softmax(scores, axis=-1)
    vh = mx.transpose(v, (1, 0, 2))
    out = probs @ vh
    out = mx.transpose(out, (1, 0, 2))
    logger.debug("swa seq=%d hq=%d window=%d out=%s", seq, hq, window, out.shape)
    return out
