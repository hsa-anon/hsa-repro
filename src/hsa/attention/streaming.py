import logging

import mlx.core as mx

from hsa.attention.latent import project_to_latent

logger = logging.getLogger(__name__)


def latent_from_sums(
    sum_k: mx.array, sum_v: mx.array, count: int, d_latent: int
) -> tuple[mx.array, mx.array]:
    """Project one completed block's running sums into its latent pair."""
    return project_to_latent(sum_k / count, d_latent), sum_v / count


def streaming_latent_attention(
    q: mx.array, k: mx.array, v: mx.array, window: int, block_size: int, d_latent: int
) -> mx.array:
    """Single-pass streaming form of the three-tier latent attention.

    Walks positions once. Maintains running per-block sums; each completed
    block is projected exactly once (including the final block, for parity
    with the batch oracle). Per query: exact window scores (d-space) +
    latent scores of completed blocks fully below the window.
    """
    seq, hq, d = q.shape
    hkv = k.shape[1]
    group = hq // hkv
    if group > 1:
        k = mx.repeat(k, group, axis=1)
        v = mx.repeat(v, group, axis=1)
    k_lat = []
    v_lat = []
    sum_k = mx.zeros((hq, d))
    sum_v = mx.zeros((hq, d))
    b_count = 0
    outs = []

    def flush() -> None:
        nonlocal sum_k, sum_v, b_count
        lk, lv = latent_from_sums(sum_k, sum_v, b_count, d_latent)
        mx.eval(lk, lv)
        k_lat.append(lk)
        v_lat.append(lv)
        sum_k = mx.zeros((hq, d))
        sum_v = mx.zeros((hq, d))
        b_count = 0

    for i in range(seq):
        if b_count == block_size:
            flush()
        sum_k = sum_k + k[i]
        sum_v = sum_v + v[i]
        b_count += 1

        ws = max(0, i - window + 1)
        wpos = list(range(ws, i + 1))
        qw = q[i]
        # (h, keys, d) @ (h, d, 1) -> (h, keys); a (1,h,d) @ (h,d,k) leading-1
        # broadcast silently duplicates heads on MLX, so batch dims must match
        kt = mx.transpose(k[wpos], (1, 0, 2))
        sw = (kt @ qw[..., None])[..., 0] / mx.sqrt(mx.array(float(d), mx.float32))
        visible = [b2 for b2 in range(len(k_lat)) if (b2 + 1) * block_size - 1 < ws]
        if visible:
            ql = project_to_latent(qw, d_latent)
            klt = mx.stack([k_lat[b2] for b2 in visible], axis=1)
            sl = (klt @ ql[..., None])[..., 0] / mx.sqrt(mx.array(float(d_latent), mx.float32))
            logits = mx.concatenate([sw, sl], axis=-1)
        else:
            logits = sw
        probs = mx.softmax(logits, axis=-1)
        pw = probs[..., : len(wpos)]
        pl = probs[..., len(wpos) :]
        o = mx.squeeze(pw[:, None, :] @ mx.transpose(v[wpos], (1, 0, 2)), axis=-2)
        if visible:
            vlv = mx.stack([v_lat[b2] for b2 in visible])
            o = o + mx.squeeze(pl[:, None, :] @ mx.transpose(vlv, (1, 0, 2)), axis=-2)
        outs.append(o[None])
    flush()
    out = mx.concatenate(outs, axis=0)
    logger.debug("streaming pass seq=%d blocks=%d", seq, len(k_lat))
    return out
