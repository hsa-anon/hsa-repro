import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hsa.torch_tier import three_tier_attention


def fp64_reference(q, k, v, window, block_size, d_latent, w_k=None, w_v=None):
    seq, h, d = q.shape
    dv = v.shape[-1]
    n_blocks = (seq + block_size - 1) // block_size
    counts = np.array([min((b + 1) * block_size, seq) - b * block_size for b in range(n_blocks)])
    pad_k = np.concatenate([k, np.zeros((n_blocks * block_size - seq, h, d))])
    pad_v = np.concatenate([v, np.zeros((n_blocks * block_size - seq, h, dv))])
    k_mean = pad_k.reshape(n_blocks, block_size, h, d).sum(axis=1) / counts[:, None, None]
    v_mean = pad_v.reshape(n_blocks, block_size, h, dv).sum(axis=1) / counts[:, None, None]
    if w_k is not None:
        k_lat = np.einsum("bhd,de->bhe", k_mean, w_k)
    else:
        k_lat = (
            k_mean[..., :d_latent]
            if d_latent <= d
            else np.concatenate([k_mean, np.zeros((*k_mean.shape[:-1], d_latent - d))], axis=-1)
        )
    if w_v is not None:
        v_lat = np.einsum("bhd,de->bhe", v_mean, w_v)
    else:
        v_lat = v_mean
    out = np.zeros((seq, h, dv))
    for i in range(seq):
        ws = max(0, i - window + 1)
        wpos = list(range(ws, i + 1))
        visible = [b for b in range(n_blocks) if (b + 1) * block_size - 1 < ws]
        landmarks = [b * block_size for b in visible]
        sw = np.einsum("hd,khd->hk", q[i], k[wpos]) / np.sqrt(d)
        parts = [sw]
        if visible:
            sl_ = np.einsum("hd,khd->hk", q[i], k[landmarks]) / np.sqrt(d)
            ql = np.einsum("hd,de->he", q[i], w_k) if w_k is not None else q[i][..., :d_latent]
            slt = np.einsum("he,bhe->hb", ql, k_lat[visible]) / np.sqrt(d_latent)
            parts += [sl_, slt]
        logits = np.concatenate(parts, axis=-1)
        p = np.exp(logits - logits.max(axis=-1, keepdims=True))
        p /= p.sum(axis=-1, keepdims=True)
        o = np.einsum("hk,khd->hd", p[:, : len(wpos)], v[wpos])
        if visible:
            n_w = len(wpos)
            o += np.einsum("hk,khd->hd", p[:, n_w : n_w + len(landmarks)], v[landmarks])
            o += np.einsum("hb,bhd->hd", p[:, n_w + len(landmarks) :], v_lat[visible])
        out[i] = o
    return out


def test_matches_fp64_reference():
    rng = np.random.default_rng(31)
    seq, h, d, window, block, d_lat = 24, 2, 16, 6, 4, 8
    q = torch.tensor(rng.standard_normal((seq, h, d)), dtype=torch.float64)
    k = torch.tensor(rng.standard_normal((seq, h, d)), dtype=torch.float64)
    v = torch.tensor(rng.standard_normal((seq, h, d)), dtype=torch.float64)
    out, _ = three_tier_attention(q, k, v, window, block, d_lat)
    ref = fp64_reference(q.numpy(), k.numpy(), v.numpy(), window, block, d_lat)
    np.testing.assert_allclose(out.detach().numpy(), ref, atol=1e-8, rtol=1e-6)


def test_landmark_self_query_not_latent_averaged():
    # query 8 sits at a block start; its output must not depend on its own
    # block's non-window content (positions 4..6), i.e. no self-averaging
    rng = np.random.default_rng(32)
    seq, h, d, window, block, d_lat = 16, 1, 8, 2, 4, 8
    q = torch.tensor(rng.standard_normal((seq, h, d)), dtype=torch.float64)
    k = torch.tensor(rng.standard_normal((seq, h, d)), dtype=torch.float64)
    v = torch.tensor(rng.standard_normal((seq, h, d)), dtype=torch.float64)
    out, _ = three_tier_attention(q, k, v, window, block, d_lat)
    base = out[8].detach().clone()
    rng2 = np.random.default_rng(99)
    k2, v2 = k.clone(), v.clone()
    k2[4:7] = torch.tensor(rng2.standard_normal((3, h, d)) * 100.0, dtype=torch.float64)
    v2[4:7] = torch.tensor(rng2.standard_normal((3, h, d)) * 100.0, dtype=torch.float64)
    out2, _ = three_tier_attention(q, k2, v2, window, block, d_lat)
    np.testing.assert_allclose(out2[8].detach().numpy(), base.numpy(), atol=1e-8)


def test_learned_projection_matches_reference():
    rng = np.random.default_rng(33)
    seq, h, d, window, block, d_lat = 18, 2, 16, 5, 4, 12
    q = torch.tensor(rng.standard_normal((seq, h, d)), dtype=torch.float64)
    k = torch.tensor(rng.standard_normal((seq, h, d)), dtype=torch.float64)
    v = torch.tensor(rng.standard_normal((seq, h, d)), dtype=torch.float64)
    w_k = torch.tensor(rng.standard_normal((d, d_lat)), dtype=torch.float64)
    w_v = torch.tensor(rng.standard_normal((d, d)), dtype=torch.float64)
    out, _ = three_tier_attention(q, k, v, window, block, d_lat, w_k=w_k, w_v=w_v)
    ref = fp64_reference(
        q.numpy(), k.numpy(), v.numpy(), window, block, d_lat, w_k.numpy(), w_v.numpy()
    )
    np.testing.assert_allclose(out.detach().numpy(), ref, atol=1e-8, rtol=1e-6)


def test_seq_not_multiple_of_block():
    rng = np.random.default_rng(34)
    seq, h, d, window, block, d_lat = 18, 2, 16, 6, 4, 8
    q = torch.tensor(rng.standard_normal((seq, h, d)), dtype=torch.float64)
    k = torch.tensor(rng.standard_normal((seq, h, d)), dtype=torch.float64)
    v = torch.tensor(rng.standard_normal((seq, h, d)), dtype=torch.float64)
    out, _ = three_tier_attention(q, k, v, window, block, d_lat)
    ref = fp64_reference(q.numpy(), k.numpy(), v.numpy(), window, block, d_lat)
    np.testing.assert_allclose(out.detach().numpy(), ref, atol=1e-8, rtol=1e-6)
