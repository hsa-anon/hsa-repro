import logging

import mlx.core as mx

logger = logging.getLogger(__name__)


def landmark_positions(seq: int, block_size: int) -> list[int]:
    return list(range(0, seq, block_size))


def landmark_attention(
    q: mx.array, k: mx.array, v: mx.array, window: int, block_size: int
) -> mx.array:
    """Two-tier attention: per-query sliding window + global landmark tokens.

    q: (seq, hq, d), k/v: (seq, hkv, d). Causal. Query qi attends positions
    [qi-window+1, qi] plus landmarks p <= qi. Landmarks are block starts.
    """
    seq, hq, d = q.shape
    hkv = k.shape[1]
    group = hq // hkv
    if group > 1:
        k = mx.repeat(k, group, axis=1)
        v = mx.repeat(v, group, axis=1)
    qh = mx.transpose(q, (1, 0, 2))
    kh = mx.transpose(k, (1, 2, 0))
    vh = mx.transpose(v, (1, 0, 2))
    scores = (qh @ kh) / mx.sqrt(mx.array(float(d), mx.float32))
    idx = mx.arange(seq)
    dist = idx[:, None] - idx[None, :]
    in_window = (dist >= 0) & (dist < window)
    lm = mx.array(landmark_positions(seq, block_size))
    is_landmark = mx.zeros((seq, seq), dtype=mx.bool_)
    is_landmark[:, lm] = True
    keep = in_window | (is_landmark & (dist >= 0))
    scores = mx.where(keep[None], scores, mx.array(-mx.inf, scores.dtype))
    probs = mx.softmax(scores, axis=-1)
    out = probs @ vh
    out = mx.transpose(out, (1, 0, 2))
    logger.debug(
        "landmark seq=%d window=%d block=%d landmarks=%d", seq, window, block_size, len(lm)
    )
    return out
