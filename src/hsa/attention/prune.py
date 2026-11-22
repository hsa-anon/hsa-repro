import logging
import math

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

# A skipped block's softmax mass is < block_size * exp(-LOGIT_GAP_EPS)
# relative to the max logit; at 20 this is negligible for any realistic
# block size.
LOGIT_GAP_EPS = 20.0


def hadamard_matrix(n: int) -> np.ndarray:
    # Walsh-Hadamard via Sylvester construction (n power of 2).
    assert n & (n - 1) == 0, "n must be a power of 2"
    h = np.array([[1.0]], dtype=np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def hadamard_rotate(x: mx.array) -> mx.array:
    d = x.shape[-1]
    h = mx.array(hadamard_matrix(d))
    return x @ h


def quantize_keys_int8(k: mx.array) -> tuple[mx.array, float]:
    # Symmetric int8 quantization of Hadamard-rotated keys, single head.
    # k: (seq, d) -> (int8 keys (seq, d), dequant scale). Rotation preserves
    # inner products with rotated queries; quantization error is absorbed by
    # the bound err term.
    kr = hadamard_rotate(k)
    scale = max(float(np.abs(np.array(kr)).max()) / 127.0, 1e-12)
    k8 = mx.round(kr / scale).astype(mx.int8)
    return k8, scale


def block_max_bounds(q: mx.array, k8: mx.array, scale: float, block_size: int) -> mx.array:
    # Per-query per-block upper bound on scaled logits (q.k / sqrt(d)) using
    # int8 keys, single head. q: (seq, d), k8: (seq, d) int8.
    # Bound: max_j (q_rot . k8_j) * scale / sqrt(d) + err, with
    # err = sqrt(d) * scale / 2 * max_i |q_rot_i| (each key element error is
    # <= scale/2; worst-case aligned over d elements).
    d = q.shape[-1]
    s = math.sqrt(d)
    qs = np.array(hadamard_rotate(q))
    k8f = np.array(k8).astype(np.float32)
    err = s * scale / 2.0 * np.abs(qs).max(axis=-1, keepdims=True)
    logits = (qs @ k8f.T) * scale / s
    seq, n_keys = logits.shape
    n_blocks = (n_keys + block_size - 1) // block_size
    pad = n_blocks * block_size - n_keys
    kt = logits.T
    if pad:
        kt = np.concatenate([kt, np.full((pad, seq), -np.inf, dtype=np.float32)], axis=0)
    blocks = kt.reshape(n_blocks, block_size, seq)
    return mx.array(blocks.max(axis=1).T + err)


def block_attention_with_skip(
    q: mx.array, k: mx.array, v: mx.array, block_size: int
) -> tuple[mx.array, int]:
    # Exact attention with per-query block skipping driven by int8 bounds.
    # Per (head, query): bounds over causally-visible keys, blocks processed
    # in bound-descending order; stop once the next bound falls below
    # (running exact max - eps). Every skipped block's max logit is then
    # <= its bound <= M - eps, so its total softmax weight is negligible
    # and the output matches dense up to float error. Only processed blocks
    # are scored exactly.
    seq, h, d = q.shape
    s = math.sqrt(d)
    qn = np.array(q)
    kn = np.array(k)
    vn = np.array(v)
    n_blocks = (seq + block_size - 1) // block_size
    out = np.zeros((seq, h, d), dtype=np.float32)
    kept = 0
    for hidx in range(h):
        k8, scale = quantize_keys_int8(k[:, hidx, :])
        k8f = np.array(k8).astype(np.float32)
        qrot = np.array(hadamard_rotate(q[:, hidx, :]))
        err = s * scale / 2.0 * np.abs(qrot).max(axis=-1)
        for i in range(seq):
            vis = i + 1
            nb_v = (vis + block_size - 1) // block_size
            dots = (qrot[i] @ k8f[:vis].T) * scale / s
            bnd = np.full(nb_v, -np.inf, dtype=np.float64)
            for b in range(nb_v):
                bnd[b] = dots[b * block_size : (b + 1) * block_size].max() + err[i]
            order = np.argsort(-bnd)
            m = -np.inf
            kept_blocks = []
            for t in range(nb_v):
                if bnd[order[t]] <= m - LOGIT_GAP_EPS:
                    break
                blk = int(order[t])
                lo = blk * block_size
                hi = min((blk + 1) * block_size, vis)
                lm = float((qn[i, hidx] @ kn[lo:hi, hidx].T).max()) / s
                m = max(m, lm)
                kept_blocks.append(blk)
            kept += len(kept_blocks)
            pos = np.concatenate(
                [np.arange(b * block_size, min((b + 1) * block_size, vis)) for b in kept_blocks]
            )
            logits = (qn[i, hidx] @ kn[pos, hidx].T) / s
            p = np.exp(logits - logits.max())
            p /= p.sum()
            out[i, hidx] = p @ vn[pos, hidx]
    logger.debug("skip attention kept=%d/%d blocks", kept, h * seq * n_blocks)
    return mx.array(out), kept
