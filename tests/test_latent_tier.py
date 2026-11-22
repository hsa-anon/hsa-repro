import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

mx.set_default_device(mx.cpu)

from hsa.attention.latent import latent_attention, project_to_latent


def test_project_to_latent_truncates_and_pads():
    rng = np.random.default_rng(6)
    x = mx.array(rng.standard_normal((4, 2, 16)).astype(np.float32))
    lat = project_to_latent(x, 8)
    assert lat.shape == (4, 2, 8)
    np.testing.assert_allclose(np.array(lat), np.array(x)[:, :, :8], atol=1e-6)
    up = project_to_latent(x, 24)
    assert up.shape == (4, 2, 24)
    np.testing.assert_allclose(np.array(up)[:, :, :16], np.array(x), atol=1e-6)
    assert float(np.abs(np.array(up)[:, :, 16:]).max()) == 0.0


def test_latent_attention_matches_reference():
    rng = np.random.default_rng(5)
    seq, hq, hkv, d, window, block, d_lat = 24, 4, 2, 16, 6, 4, 8
    q = mx.array(rng.standard_normal((seq, hq, d)).astype(np.float32))
    k = mx.array(rng.standard_normal((seq, hkv, d)).astype(np.float32))
    v = mx.array(rng.standard_normal((seq, hkv, d)).astype(np.float32))
    out = np.array(latent_attention(q, k, v, window, block, d_lat))

    kx = np.repeat(np.array(k), hq // hkv, axis=1)
    vx = np.repeat(np.array(v), hq // hkv, axis=1)
    n_blocks = (seq + block - 1) // block
    k_lat = np.zeros((n_blocks, hq, d_lat), dtype=np.float32)
    v_lat = np.zeros((n_blocks, hq, d), dtype=np.float32)
    for b in range(n_blocks):
        seg = slice(b * block, min((b + 1) * block, seq))
        k_lat[b] = project_to_latent_np(kx[seg].mean(axis=0), d_lat)
        v_lat[b] = vx[seg].mean(axis=0)

    ref = np.zeros((seq, hq, d), dtype=np.float32)
    for qi in range(seq):
        wpos = list(range(max(0, qi - window + 1), qi + 1))
        sw = np.einsum("hd,hpd->hp", np.array(q)[qi], kx[wpos].transpose(1, 0, 2)) / np.sqrt(d)
        b_start = max(0, (qi - window + 1 + block - 1) // block)
        lpos = [
            b for b in range(b_start) if min((b + 1) * block, seq) - 1 < max(0, qi - window + 1)
        ]
        sl = (
            np.einsum(
                "hd,hbd->hb",
                project_to_latent_np(np.array(q)[qi], d_lat),
                k_lat[lpos].transpose(1, 0, 2),
            )
            / np.sqrt(d_lat)
            if lpos
            else np.zeros((hq, 0), dtype=np.float32)
        )
        logits = np.concatenate([sw, sl], axis=-1)
        logits = np.where(np.isfinite(logits), logits, -np.inf)
        p = np.exp(logits - logits.max(axis=-1, keepdims=True))
        p = p / p.sum(axis=-1, keepdims=True)
        pw, pl = p[:, : len(wpos)], p[:, len(wpos) :]
        ref[qi] = np.einsum("hp,hpd->hd", pw, vx[wpos].transpose(1, 0, 2))
        if lpos:
            ref[qi] += np.einsum("hb,hbd->hd", pl, v_lat[lpos].transpose(1, 0, 2))
    np.testing.assert_allclose(out, ref, atol=1e-4, rtol=1e-4)


def project_to_latent_np(x, d_lat):
    if d_lat <= x.shape[-1]:
        return x[..., :d_lat]
    return np.concatenate(
        [x, np.zeros((*x.shape[:-1], d_lat - x.shape[-1]), dtype=x.dtype)], axis=-1
    )
