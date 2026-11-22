import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hsa.torch_tier_fast import three_tier_attention_fast


def test_torch_chunked_prefill_matches_one_shot():
    # Chunked run (q_offset = chunk start, k/v = full history so far) must
    # equal the one-shot run on both chunk row ranges - mirrors the MLX
    # kernel test and pins the serving/eval chunking contract.
    rng = np.random.default_rng(31)
    B, T, Hq, Hkv, D, Dv, window, block_size, L = 1, 64, 4, 2, 16, 16, 8, 4, 32
    q = rng.standard_normal((B, T, Hq, D)).astype(np.float32)
    k = rng.standard_normal((B, T, Hkv, Dv)).astype(np.float32)
    v = rng.standard_normal((B, T, Hkv, Dv)).astype(np.float32)
    w_q = rng.standard_normal((Hq * D, L)).astype(np.float32) * 0.02
    w_k = rng.standard_normal((Hkv * Dv, L)).astype(np.float32) * 0.02
    tq, tk, tv = torch.tensor(q), torch.tensor(k), torch.tensor(v)
    tw_q, tw_k = torch.tensor(w_q), torch.tensor(w_k)
    one_shot, _ = three_tier_attention_fast(tq, tk, tv, window, block_size, tw_q, tw_k)
    cut = 40
    c1, _ = three_tier_attention_fast(
        tq[:, :cut], tk[:, :cut], tv[:, :cut], window, block_size, tw_q, tw_k
    )
    c2, _ = three_tier_attention_fast(
        tq[:, cut:], tk, tv, window, block_size, tw_q, tw_k, q_offset=cut
    )
    np.testing.assert_allclose(c1[:, :cut].numpy(), one_shot[:, :cut].numpy(), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(
        c2[:, : T - cut].numpy(), one_shot[:, cut:].numpy(), atol=1e-4, rtol=1e-4
    )
