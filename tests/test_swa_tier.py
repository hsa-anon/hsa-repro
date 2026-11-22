import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

# fp32-exact parity: run MLX on CPU (Metal matmul quantizes to ~6e-4)
mx.set_default_device(mx.cpu)

from hsa.attention.window import sliding_window_attention


def dense_reference(q, k, v, window):
    seq, hq, d = q.shape
    kv_heads = k.shape[1]
    group = hq // kv_heads
    kx = np.repeat(k, group, axis=1)
    vx = np.repeat(v, group, axis=1)
    scores = np.einsum("qhd,khd->hqk", q, kx) / np.sqrt(d)
    mask = np.full((seq, seq), -np.inf)
    for qi in range(seq):
        lo = max(0, qi - window + 1)
        mask[qi, lo : qi + 1] = 0.0
    probs = np.exp(
        scores + mask[np.newaxis] - np.max(scores + mask[np.newaxis], axis=-1, keepdims=True)
    )
    probs = probs / probs.sum(axis=-1, keepdims=True)
    return np.einsum("hqk,khd->qhd", probs, vx)


def test_swa_matches_dense_reference():
    rng = np.random.default_rng(0)
    seq, hq, hkv, d, window = 32, 4, 2, 16, 8
    q = mx.array(rng.standard_normal((seq, hq, d)).astype(np.float32))
    k = mx.array(rng.standard_normal((seq, hkv, d)).astype(np.float32))
    v = mx.array(rng.standard_normal((seq, hkv, d)).astype(np.float32))
    out = np.array(sliding_window_attention(q, k, v, window))
    ref = dense_reference(np.array(q), np.array(k), np.array(v), window)
    np.testing.assert_allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_swa_first_token_is_identity_of_self():
    rng = np.random.default_rng(1)
    seq, hq, hkv, d, window = 8, 2, 2, 8, 4
    q = mx.array(rng.standard_normal((seq, hq, d)).astype(np.float32))
    k = mx.array(rng.standard_normal((seq, hkv, d)).astype(np.float32))
    v = mx.array(rng.standard_normal((seq, hkv, d)).astype(np.float32))
    out = np.array(sliding_window_attention(q, k, v, window))
    np.testing.assert_allclose(out[0], np.array(v)[0], atol=1e-5)
