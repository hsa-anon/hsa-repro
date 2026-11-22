import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")

mx.set_default_device(mx.cpu)

from hsa.attention.latent import latent_attention
from hsa.attention.streaming import streaming_latent_attention


def torch_three_tier_oracle(q, k, v, window, block_size, d_latent):
    """Independent torch oracle for the latent tier contract."""
    seq, hq, d = q.shape
    hkv = k.shape[1]
    group = hq // hkv
    kr = np.repeat(k, group, axis=1)
    vr = np.repeat(v, group, axis=1)
    n_blocks = (seq + block_size - 1) // block_size
    counts = np.array([min((b + 1) * block_size, seq) - b * block_size for b in range(n_blocks)])
    pad = np.concatenate([kr, np.zeros((n_blocks * block_size - seq, hq, d))])
    pv = np.concatenate([vr, np.zeros((n_blocks * block_size - seq, hq, v.shape[-1]))])
    kb = pad.reshape(n_blocks, block_size, hq, d).sum(axis=1) / counts[:, None, None]
    vb = pv.reshape(n_blocks, block_size, hq, -1).sum(axis=1) / counts[:, None, None]
    k_lat = kb[..., :d_latent]
    outs = []
    for i in range(seq):
        ws = max(0, i - window + 1)
        wpos = list(range(ws, i + 1))
        sw = np.einsum("hd,khd->hk", q[i], kr[wpos]) / np.sqrt(d)
        visible = [b for b in range(n_blocks) if (b + 1) * block_size - 1 < ws]
        if visible:
            sl = np.einsum("hd,bhd->hb", q[i][..., :d_latent], k_lat[visible]) / np.sqrt(d_latent)
            logits = np.concatenate([sw, sl], axis=-1)
        else:
            logits = sw
        probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        pw, pl = probs[:, : len(wpos)], probs[:, len(wpos) :]
        o = np.einsum("hk,khd->hd", pw, vr[wpos])
        if visible:
            o += np.einsum("hb,bhd->hd", pl, vb[visible])
        outs.append(o)
    return np.stack(outs)


@pytest.mark.parametrize(
    "seq,hq,hkv,d,window,block,d_lat",
    [
        (24, 4, 2, 16, 6, 4, 8),
        (18, 2, 2, 16, 20, 4, 8),  # window > seq, no GQA repeat
        (18, 4, 2, 8, 5, 4, 12),  # seq % block != 0, d_latent > d (pad path)
    ],
)
@pytest.mark.parametrize("impl", [latent_attention, streaming_latent_attention])
def test_torch_oracle_parity(impl, seq, hq, hkv, d, window, block, d_lat):
    rng = np.random.default_rng(21)
    q = rng.standard_normal((seq, hq, d)).astype(np.float32)
    k = rng.standard_normal((seq, hkv, d)).astype(np.float32)
    v = rng.standard_normal((seq, hkv, d)).astype(np.float32)
    out = np.array(impl(mx.array(q), mx.array(k), mx.array(v), window, block, d_lat))
    ref = torch_three_tier_oracle(
        torch.tensor(q), torch.tensor(k), torch.tensor(v), window, block, d_lat
    )
    np.testing.assert_allclose(out, ref, atol=1e-4, rtol=1e-4)
