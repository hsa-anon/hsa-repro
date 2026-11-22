import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

mx.set_default_device(mx.cpu)

from hsa.attention.prune import (
    block_attention_with_skip,
    block_max_bounds,
    hadamard_rotate,
    quantize_keys_int8,
)


def test_hadamard_rotation_preserves_inner_products():
    rng = np.random.default_rng(9)
    x = mx.array(rng.standard_normal((6, 16)).astype(np.float32))
    y = mx.array(rng.standard_normal((6, 16)).astype(np.float32))
    xr, yr = hadamard_rotate(x), hadamard_rotate(y)
    dots_before = np.array(x) @ np.array(y).T
    dots_after = np.array(xr) @ np.array(yr).T
    np.testing.assert_allclose(dots_before, dots_after, atol=1e-4, rtol=1e-4)


def test_quantize_returns_int8_dtype():
    rng = np.random.default_rng(10)
    k = mx.array(rng.standard_normal((32, 16)).astype(np.float32))
    k8, scale = quantize_keys_int8(k)
    assert np.array(k8).dtype == np.int8
    assert scale > 0
    deq = np.array(k8).astype(np.float32) * scale
    np.testing.assert_allclose(deq, np.array(hadamard_rotate(k)), atol=scale / 2 + 1e-6)


def test_quantized_bound_covers_true_max():
    rng = np.random.default_rng(10)
    seq, d, block = 32, 16, 8
    q = rng.standard_normal((seq, d)).astype(np.float32)
    k = rng.standard_normal((seq, d)).astype(np.float32)
    k8, scale = quantize_keys_int8(mx.array(k))
    bounds = np.array(block_max_bounds(mx.array(q), k8, scale, block))
    true_max = (np.array(q) @ np.array(k).T / np.sqrt(d)).reshape(seq, -1, block).max(axis=-1)
    assert np.all(bounds + 1e-4 >= true_max), "bound violated - pruning would be unsound"


def dense_reference(q, k, v):
    seq, _h, d = q.shape
    scores = np.einsum("qhd,khd->hqk", q, k) / np.sqrt(d)
    mask = np.triu(np.ones((seq, seq), dtype=bool), 1)
    scores = np.where(mask[None], -np.inf, scores)
    p = np.exp(scores - scores.max(axis=-1, keepdims=True))
    p = p / p.sum(axis=-1, keepdims=True)
    return np.einsum("hqk,khd->qhd", p, v)


def test_skip_output_matches_dense():
    rng = np.random.default_rng(11)
    seq, h, d, block = 32, 2, 16, 8
    q = mx.array(rng.standard_normal((seq, h, d)).astype(np.float32))
    k = mx.array(rng.standard_normal((seq, h, d)).astype(np.float32))
    v = mx.array(rng.standard_normal((seq, h, d)).astype(np.float32))
    out_skip, n_kept = block_attention_with_skip(q, k, v, block)
    ref = dense_reference(np.array(q), np.array(k), np.array(v))
    np.testing.assert_allclose(np.array(out_skip), ref, atol=1e-4, rtol=1e-4)
    valid = sum((i + 1 + block - 1) // block for i in range(seq))
    assert 0 < n_kept <= h * valid, "kept count must respect causal visibility"


def test_skip_drops_distant_low_score_block():
    rng = np.random.default_rng(12)
    seq, h, d, block = 32, 1, 16, 8
    q = mx.array((rng.standard_normal((seq, h, d)) * 3.5).astype(np.float32))
    k = mx.array((rng.standard_normal((seq, h, d)) * 3.5).astype(np.float32))
    k = mx.array(np.concatenate([np.zeros((block, h, d), dtype=np.float32), np.array(k)[block:]]))
    v = mx.array(rng.standard_normal((seq, h, d)).astype(np.float32))
    out_skip, n_kept = block_attention_with_skip(q, k, v, block)
    ref = dense_reference(np.array(q), np.array(k), np.array(v))
    np.testing.assert_allclose(np.array(out_skip), ref, atol=1e-4, rtol=1e-4)
    valid = sum((i + 1 + block - 1) // block for i in range(seq))
    assert n_kept < h * valid, "zero-score distant block must be skipped for most queries"
