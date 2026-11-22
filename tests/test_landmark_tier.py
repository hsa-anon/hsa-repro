import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

mx.set_default_device(mx.cpu)

from hsa.attention.landmark import landmark_attention


def dense_reference_window_plus_landmarks(q, k, v, window, block_size, seq):
    seq_len, hq, d = q.shape
    kv_heads = k.shape[1]
    group = hq // kv_heads
    kx = np.repeat(np.array(k), group, axis=1)
    vx = np.repeat(np.array(v), group, axis=1)
    landmark_pos = list(range(0, seq_len, block_size))
    out = np.zeros((seq_len, hq, d), dtype=np.float32)
    for qi in range(seq_len):
        allowed = set(range(max(0, qi - window + 1), qi + 1)) | {p for p in landmark_pos if p <= qi}
        pos = sorted(allowed)
        scores = np.einsum("hd,hpd->hp", np.array(q)[qi], kx[pos].transpose(1, 0, 2)) / np.sqrt(d)
        probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs = probs / probs.sum(axis=-1, keepdims=True)
        out[qi] = np.einsum("hp,hpd->hd", probs, vx[pos].transpose(1, 0, 2))
    return out


def test_landmark_tier_matches_reference():
    rng = np.random.default_rng(3)
    seq, hq, hkv, d, window, block = 32, 4, 2, 16, 8, 4
    q = mx.array(rng.standard_normal((seq, hq, d)).astype(np.float32))
    k = mx.array(rng.standard_normal((seq, hkv, d)).astype(np.float32))
    v = mx.array(rng.standard_normal((seq, hkv, d)).astype(np.float32))
    out = np.array(landmark_attention(q, k, v, window, block))
    ref = dense_reference_window_plus_landmarks(q, k, v, window, block, seq)
    np.testing.assert_allclose(out, ref, atol=1e-4, rtol=1e-4)


def test_landmark_positions_are_block_starts():
    from hsa.attention.landmark import landmark_positions

    assert landmark_positions(10, 4) == [0, 4, 8]
    assert landmark_positions(4, 4) == [0]
