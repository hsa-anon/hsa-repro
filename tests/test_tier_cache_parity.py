import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hsa.eval.tier_cache import TierLayerCache
from hsa.torch_tier_fast import three_tier_attention_fast

torch.manual_seed(7)


def test_parity():
    T, Hq, Hkv, D, window, block, L = 300, 6, 2, 32, 16, 64, 8
    q = torch.randn(1, T, Hq, D)
    k = torch.randn(1, T, Hkv, D)
    v = torch.randn(1, T, Hkv, D)
    w_q_lat = torch.randn(Hq * D, L) * 0.05
    w_k_lat = torch.randn(Hkv * D, L) * 0.05

    out_full, _ = three_tier_attention_fast(
        q, k, v, window, block, w_q_lat=w_q_lat, w_k_lat=w_k_lat
    )

    T0 = 100
    cache = TierLayerCache(window, block)
    cache.seed(k[0, :T0], v[0, :T0])
    for t in range(T0, T):
        o = cache.decode(q[0, t, :, :], k[0, t], v[0, t], w_q_lat, w_k_lat)
        ref = out_full[0, t]
        assert torch.allclose(o, ref, atol=1e-4), (
            f"mismatch at t={t}: max={float((o - ref).abs().max())}"
        )
    print("parity OK: decode matches kernel rows T0..T")


def test_short_prompt():
    T, Hq, Hkv, D, window, block, L = 50, 6, 2, 32, 16, 64, 8
    q = torch.randn(1, T, Hq, D)
    k = torch.randn(1, T, Hkv, D)
    v = torch.randn(1, T, Hkv, D)
    w_q_lat = torch.randn(Hq * D, L) * 0.05
    w_k_lat = torch.randn(Hkv * D, L) * 0.05

    out_full, _ = three_tier_attention_fast(
        q, k, v, window, block, w_q_lat=w_q_lat, w_k_lat=w_k_lat
    )

    T0 = 10
    cache = TierLayerCache(window, block)
    cache.seed(k[0, :T0], v[0, :T0])
    for t in range(T0, T):
        o = cache.decode(q[0, t], k[0, t], v[0, t], w_q_lat, w_k_lat)
        ref = out_full[0, t]
        assert torch.allclose(o, ref, atol=1e-4), (
            f"mismatch at t={t}: max={float((o - ref).abs().max())}"
        )
    print("short-prompt parity OK")


if __name__ == "__main__":
    test_parity()
    test_short_prompt()


def test_topk_decode_matches_dense_attention():
    # topk retrieval with K >= T must equal exact softmax attention per
    # (kv-head, group) pair; pins the retrieval branch against an oracle.
    import torch

    from hsa.eval.tier_cache import TierLayerCache

    torch.manual_seed(7)
    T, Hkv, group, D, Dv, L = 50, 2, 3, 16, 8, 32
    Hq = Hkv * group
    k = torch.randn(T, Hkv, D)
    v = torch.randn(T, Hkv, Dv)
    q = torch.randn(Hq, D)
    cache = TierLayerCache(window=8, block_size=16)
    cache.seed(
        k[:-1], v[:-1]
    )  # decode() appends the new token itself; seeding all T rows would double-count k[-1]
    out = cache.decode(
        q, k[-1], v[-1], torch.randn(Hkv * Dv, L), torch.randn(Hkv * Dv, L), topk=T + 5
    )
    ref = torch.zeros(Hq, Dv)
    for h in range(Hkv):
        for g in range(group):
            qh = q[h * group + g]
            probs = torch.softmax(k[:, h] @ qh / D**0.5, dim=0)
            ref[h * group + g] = probs.T @ v[:, h]
    assert torch.allclose(out, ref, atol=1e-4), f"max diff {(out - ref).abs().max()}"


def test_topk_small_retrieves_needle():
    # With topk smaller than T the needle key must still dominate the
    # output when its score is far above the filler keys'.
    import torch

    from hsa.eval.tier_cache import TierLayerCache

    torch.manual_seed(3)
    T, Hkv, group, D, Dv, L = 60, 1, 2, 16, 8, 32
    k = torch.randn(T, Hkv, D) * 0.1
    v = torch.randn(T, Hkv, Dv) * 0.1
    needle_pos = 5  # far outside any small window
    needle_v = torch.tensor([[3.0] * Dv])
    k[needle_pos] = torch.randn(
        D
    )  # full-scale key so the aligned query score dominates all filler scores
    v[needle_pos] = needle_v
    q = k[needle_pos].reshape(1, D).repeat(group, 1)  # query aligns with needle key
    cache = TierLayerCache(window=8, block_size=16)
    cache.seed(k[:-1], v[:-1])  # decode() appends the new token itself
    out = cache.decode(q, k[-1], v[-1], torch.randn(Hkv * Dv, L), torch.randn(Hkv * Dv, L), topk=4)
    assert out[0].dist(needle_v[0]) < 0.5, f"needle lost: {out[0][:3]}"
