import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")

mx.set_default_device(mx.cpu)

from hsa.attention.three_tier_mlx import three_tier_attention_fast_mlx
from hsa.torch_tier_fast import three_tier_attention_fast


def _run_case(T, Hq, Hkv, D, Dv, window, block_size, L, seed):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((T, Hq, D)).astype(np.float32)
    k = rng.standard_normal((T, Hkv, Dv)).astype(np.float32)
    v = rng.standard_normal((T, Hkv, Dv)).astype(np.float32)
    w_q = rng.standard_normal((Hq * D, L)).astype(np.float32) * 0.02
    w_k = rng.standard_normal((Hkv * Dv, L)).astype(np.float32) * 0.02
    qt = torch.tensor(q)[None]
    kt = torch.tensor(k)[None]
    vt = torch.tensor(v)[None]
    ref, _ = three_tier_attention_fast(
        qt, kt, vt, window, block_size, torch.tensor(w_q), torch.tensor(w_k)
    )
    got = three_tier_attention_fast_mlx(
        mx.array(q), mx.array(k), mx.array(v), window, block_size, mx.array(w_q), mx.array(w_k)
    )
    np.testing.assert_allclose(np.array(got), ref[0].numpy(), atol=1e-4, rtol=1e-4)


def test_mlx_matches_torch_oracle_no_latent():
    _run_case(37, 6, 2, 16, 16, 8, 4, 0, seed=11)


def test_mlx_matches_torch_oracle_with_latent():
    _run_case(37, 6, 2, 16, 16, 8, 4, 32, seed=12)


def test_mlx_matches_torch_oracle_gqa6_uneven():
    # serving-like head layout (group 6) and a T not divisible by block size
    _run_case(53, 6, 1, 8, 8, 6, 5, 24, seed=13)


def test_mlx_matches_torch_oracle_latent_clamp():
    rng = np.random.default_rng(14)
    T, Hq, Hkv, D, Dv, window, block_size, L = 40, 4, 2, 16, 16, 8, 4, 32
    q = rng.standard_normal((T, Hq, D)).astype(np.float32)
    k = rng.standard_normal((T, Hkv, Dv)).astype(np.float32)
    v = rng.standard_normal((T, Hkv, Dv)).astype(np.float32)
    w_q = rng.standard_normal((Hq * D, L)).astype(np.float32) * 0.5
    w_k = rng.standard_normal((Hkv * Dv, L)).astype(np.float32) * 0.5
    ref, _ = three_tier_attention_fast(
        torch.tensor(q)[None],
        torch.tensor(k)[None],
        torch.tensor(v)[None],
        window,
        block_size,
        torch.tensor(w_q),
        torch.tensor(w_k),
        latent_clamp=2.0,
    )
    got = three_tier_attention_fast_mlx(
        mx.array(q),
        mx.array(k),
        mx.array(v),
        window,
        block_size,
        mx.array(w_q),
        mx.array(w_k),
        latent_clamp=2.0,
    )
    np.testing.assert_allclose(np.array(got), ref[0].numpy(), atol=1e-4, rtol=1e-4)


def test_mlx_chunked_prefill_matches_one_shot():
    # Serving property: feeding the history in two chunks (q_offset = chunk
    # start, k/v = full history so far) must equal the one-shot run on the
    # second chunk's rows - chunking only changes which queries run, never
    # what any query sees.
    rng = np.random.default_rng(15)
    T, Hq, Hkv, D, Dv, window, block_size, L = 64, 4, 2, 16, 16, 8, 4, 32
    q = rng.standard_normal((T, Hq, D)).astype(np.float32)
    k = rng.standard_normal((T, Hkv, Dv)).astype(np.float32)
    v = rng.standard_normal((T, Hkv, Dv)).astype(np.float32)
    w_q = rng.standard_normal((Hq * D, L)).astype(np.float32) * 0.02
    w_k = rng.standard_normal((Hkv * Dv, L)).astype(np.float32) * 0.02
    one_shot = three_tier_attention_fast_mlx(
        mx.array(q),
        mx.array(k),
        mx.array(v),
        window,
        block_size,
        mx.array(w_q),
        mx.array(w_k),
    )
    cut = 40
    c1 = three_tier_attention_fast_mlx(
        mx.array(q[:cut]),
        mx.array(k[:cut]),
        mx.array(v[:cut]),
        window,
        block_size,
        mx.array(w_q),
        mx.array(w_k),
    )
    c2 = three_tier_attention_fast_mlx(
        mx.array(q[cut:]),
        mx.array(k),
        mx.array(v),
        window,
        block_size,
        mx.array(w_q),
        mx.array(w_k),
        q_offset=cut,
    )
    np.testing.assert_allclose(np.array(c1), np.array(one_shot[:cut]), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.array(c2), np.array(one_shot[cut:]), atol=1e-4, rtol=1e-4)
