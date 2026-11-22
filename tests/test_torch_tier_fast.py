import numpy as np
import pytest
import torch

from hsa.torch_tier_fast import three_tier_attention_fast


def fp64_reference(q, k, v, window, block_size, w_q_lat=None, w_k_lat=None):
    B, T, Hq, D = q.shape
    Hkv = k.shape[2]
    Dv = v.shape[-1]
    group = Hq // Hkv
    L = w_k_lat.shape[-1] if w_k_lat is not None else 0
    n_blocks = (T + block_size - 1) // block_size
    counts = np.array([min((b + 1) * block_size, T) - b * block_size for b in range(n_blocks)])
    pad = n_blocks * block_size - T
    kp = np.concatenate([k, np.zeros((B, pad, Hkv, Dv))], axis=1)
    vp = np.concatenate([v, np.zeros((B, pad, Hkv, Dv))], axis=1)
    k_mean = kp.reshape(B, n_blocks, block_size, Hkv, Dv).sum(axis=2) / counts[None, :, None, None]
    v_mean = vp.reshape(B, n_blocks, block_size, Hkv, Dv).sum(axis=2) / counts[None, :, None, None]
    np.repeat(k, group, axis=2)
    np.repeat(v, group, axis=2)
    krr = np.repeat(kp, group, axis=2)
    vrr = np.repeat(vp, group, axis=2)
    out = np.zeros((B, T, Hq, Dv))
    for b in range(B):
        for i in range(T):
            ws = max(0, i - window + 1)
            wpos = [p for p in range(max(0, ws), i + 1)]
            visible = [bl for bl in range(n_blocks) if (bl + 1) * block_size - 1 < ws]
            landmarks = [bl * block_size for bl in visible]
            sw = np.einsum("hd,khd->hk", q[b, i], krr[b, wpos]) / np.sqrt(D)
            parts = [sw]
            if visible:
                sl_ = np.einsum("hd,khd->hk", q[b, i], krr[b, landmarks]) / np.sqrt(D)
                parts.append(sl_)
            if L and visible:
                q_lat = q[b, i].reshape(Hq * D) @ w_q_lat
                k_lat = k_mean[b, visible].reshape(len(visible), Hkv * Dv) @ w_k_lat
                slt_shared = (q_lat @ k_lat.T) / np.sqrt(L)
                parts.append(np.tile(slt_shared, (Hq, 1)))
            logits = np.concatenate(parts, axis=-1)
            p = np.exp(logits - logits.max(axis=-1, keepdims=True))
            p /= p.sum(axis=-1, keepdims=True)
            o = np.einsum("hk,khd->hd", p[:, : len(wpos)], vrr[b, wpos])
            if visible:
                n_w = len(wpos)
                o += np.einsum("hk,khd->hd", p[:, n_w : n_w + len(landmarks)], vrr[b, landmarks])
                if L:
                    o += np.einsum(
                        "hk,khd->hd",
                        p[:, n_w + len(landmarks) :],
                        np.repeat(v_mean[b, visible], group, axis=1),
                    )
            out[b, i] = o
    return out


@pytest.mark.parametrize("use_latent", [False, True])
def test_fast_matches_fp64_reference(use_latent):
    torch.manual_seed(3)
    B, T, Hq, Hkv, D, Dv, window, block = 1, 18, 4, 2, 8, 8, 6, 4
    L = 32
    q = torch.randn(B, T, Hq, D, dtype=torch.float64)
    k = torch.randn(B, T, Hkv, Dv, dtype=torch.float64)
    v = torch.randn(B, T, Hkv, Dv, dtype=torch.float64)
    if use_latent:
        w_q_lat = torch.randn(Hq * D, L, dtype=torch.float64) * 0.1
        w_k_lat = torch.randn(Hkv * Dv, L, dtype=torch.float64) * 0.1
    else:
        w_q_lat = w_k_lat = None
    out, _ = three_tier_attention_fast(q, k, v, window, block, w_q_lat=w_q_lat, w_k_lat=w_k_lat)
    ref = fp64_reference(
        q.numpy(),
        k.numpy(),
        v.numpy(),
        window,
        block,
        w_q_lat.numpy() if use_latent else None,
        w_k_lat.numpy() if use_latent else None,
    )
    np.testing.assert_allclose(out.detach().numpy(), ref, atol=1e-8, rtol=1e-6)


def test_fast_batched_matches_single():
    torch.manual_seed(4)
    B, T, Hq, Hkv, D, window, block, L = 2, 16, 4, 2, 8, 6, 4, 16
    q = torch.randn(B, T, Hq, D, dtype=torch.float64)
    k = torch.randn(B, T, Hkv, D, dtype=torch.float64)
    v = torch.randn(B, T, Hkv, D, dtype=torch.float64)
    w_q_lat = torch.randn(Hq * D, L, dtype=torch.float64) * 0.1
    w_k_lat = torch.randn(Hkv * D, L, dtype=torch.float64) * 0.1
    out_b, _ = three_tier_attention_fast(q, k, v, window, block, w_q_lat=w_q_lat, w_k_lat=w_k_lat)
    out_s, _ = three_tier_attention_fast(
        q[:1], k[:1], v[:1], window, block, w_q_lat=w_q_lat, w_k_lat=w_k_lat
    )
    np.testing.assert_allclose(
        out_b[0].detach().numpy(), out_s[0].detach().numpy(), atol=1e-8, rtol=1e-6
    )


def test_fast_gradients_flow_to_latent_weights():
    torch.manual_seed(5)
    B, T, Hq, Hkv, D, window, block, L = 1, 12, 2, 2, 8, 4, 4, 16
    q = torch.randn(B, T, Hq, D)
    k = torch.randn(B, T, Hkv, D, requires_grad=True)
    v = torch.randn(B, T, Hkv, D, requires_grad=True)
    # asymmetric init (matches real design): tiny-random query proj, zero key proj
    w_q_lat = (torch.randn(Hq * D, L) * 1e-3).requires_grad_(True)
    w_k_lat = torch.zeros(Hkv * D, L, requires_grad=True)
    out, _ = three_tier_attention_fast(q, k, v, window, block, w_q_lat=w_q_lat, w_k_lat=w_k_lat)
    out.sum().backward()
    assert w_k_lat.grad is not None and w_k_lat.grad.abs().sum() > 0
    # w_q_lat gradient flows through k_lat; it lights up once w_k_lat != 0
    w_k_lat2 = torch.randn(Hkv * D, L) * 0.1
    w_q_lat2 = (torch.randn(Hq * D, L) * 1e-3).requires_grad_(True)
    out2, _ = three_tier_attention_fast(q, k, v, window, block, w_q_lat=w_q_lat2, w_k_lat=w_k_lat2)
    out2.sum().backward()
    assert w_q_lat2.grad is not None and w_q_lat2.grad.abs().sum() > 0
