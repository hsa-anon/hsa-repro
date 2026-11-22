import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

mx.set_default_device(mx.cpu)

from hsa.attention.latent import latent_attention
from hsa.attention.streaming import streaming_latent_attention


def test_streaming_matches_batch():
    rng = np.random.default_rng(13)
    seq, hq, hkv, d, window, block, d_lat = 24, 4, 2, 16, 6, 4, 8
    q = mx.array(rng.standard_normal((seq, hq, d)).astype(np.float32))
    k = mx.array(rng.standard_normal((seq, hkv, d)).astype(np.float32))
    v = mx.array(rng.standard_normal((seq, hkv, d)).astype(np.float32))
    out_stream = np.array(streaming_latent_attention(q, k, v, window, block, d_lat))
    out_batch = np.array(latent_attention(q, k, v, window, block, d_lat))
    np.testing.assert_allclose(out_stream, out_batch, atol=1e-4, rtol=1e-4)


def test_streaming_is_single_pass():
    from hsa.attention import streaming

    calls = {"n": 0}
    orig = streaming.latent_from_sums
    streaming.latent_from_sums = lambda sk, sv, c, d: (
        calls.__setitem__("n", calls["n"] + 1),
        orig(sk, sv, c, d),
    )[1]
    try:
        rng = np.random.default_rng(14)
        seq, hq, hkv, d, window, block, d_lat = 16, 2, 2, 8, 4, 4, 4
        q = mx.array(rng.standard_normal((seq, hq, d)).astype(np.float32))
        k = mx.array(rng.standard_normal((seq, hkv, d)).astype(np.float32))
        v = mx.array(rng.standard_normal((seq, hkv, d)).astype(np.float32))
        streaming.streaming_latent_attention(q, k, v, window, block, d_lat)
    finally:
        streaming.latent_from_sums = orig
    # one projection per completed block, not one per query
    assert calls["n"] == seq // block, f"expected {seq // block} projections, got {calls['n']}"
