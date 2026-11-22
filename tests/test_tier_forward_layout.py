import torch

from hsa.train.qwen3_tier import make_tier_forward


class _StubAttn(torch.nn.Module):
    # Minimal Qwen3-style full-attention module: non-gated, no decode mode -
    # exercises the plain full-seq training branch of tier_forward.
    def __init__(self, hq=4, hkv=2, d=8, seq=8):
        super().__init__()
        self.hq, self.hkv, self.d = hq, hkv, d
        self.q_proj = torch.nn.Linear(d, hq * d, bias=False)
        self.k_proj = torch.nn.Linear(d, hkv * d, bias=False)
        self.v_proj = torch.nn.Linear(d, hkv * d, bias=False)
        self.o_proj = torch.nn.Linear(hq * d, d, bias=False)
        self.q_norm = torch.nn.Identity()
        self.k_norm = torch.nn.Identity()
        self.config = type("C", (), {"num_attention_heads": hq, "num_key_value_heads": hkv})()
        self.head_dim = d
        self.tier_window = 4
        self.tier_block = 4
        self.w_q_lat = torch.nn.Parameter(torch.randn(hq * d, 16) * 1e-3)
        self.w_k_lat = torch.nn.Parameter(torch.zeros(hkv * d, 16))
        self._tier_decode = False
        self.forward = make_tier_forward(
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("dense fallback called"))
        ).__get__(self)


def test_tier_forward_full_seq_training_branch():
    # Regression: the non-decode branch passed key/value states in HF layout
    # (B, Hkv, T, D) while three_tier_attention_fast expects (B, T, Hkv, D) -
    # Hkv read from k.shape[2] became T and group became 0, so the grouped
    # view crashed ("shape '[1, T, T, 0, D]' is invalid").
    torch.manual_seed(0)
    attn = _StubAttn()
    x = torch.randn(1, 8, 8)
    cos = torch.ones(1, 8, 8)
    sin = torch.zeros(1, 8, 8)
    out, _ = attn(x, (cos, sin))
    assert out.shape == (1, 8, 8)
    assert torch.isfinite(out).all()


def test_exact_tiers_detached_latent_grads_flow():
    # Exact window/landmark tiers carry no trainable params (backbone frozen);
    # they are computed under no_grad so the 8k window-gather graph never
    # materializes. The trainable w_q_lat/w_k_lat sit on the latent path and
    # must still receive gradients, and q must receive its gradient through
    # the latent path (layer inputs of earlier tier layers).
    from hsa.torch_tier_fast import three_tier_attention_fast

    torch.manual_seed(1)
    B, T, Hq, Hkv, D, Dv, window, block, L = 1, 12, 4, 2, 8, 8, 6, 4, 16
    q = torch.randn(B, T, Hq, D, dtype=torch.float64, requires_grad=True)
    k = torch.randn(B, T, Hkv, Dv, dtype=torch.float64, requires_grad=True)
    v = torch.randn(B, T, Hkv, Dv, dtype=torch.float64, requires_grad=True)
    w_q_lat = torch.randn(Hq * D, L, dtype=torch.float64) * 0.1
    w_k_lat = torch.randn(Hkv * Dv, L, dtype=torch.float64) * 0.1
    w_q_lat.requires_grad_(True)
    w_k_lat.requires_grad_(True)
    out, _ = three_tier_attention_fast(q, k, v, window, block, w_q_lat=w_q_lat, w_k_lat=w_k_lat)
    assert out.requires_grad
    out.sum().backward()
    assert w_q_lat.grad is not None and torch.isfinite(w_q_lat.grad).all()
    assert w_q_lat.grad.abs().sum() > 0
    assert w_k_lat.grad is not None and torch.isfinite(w_k_lat.grad).all()
    assert w_k_lat.grad.abs().sum() > 0
    # backbone grad flows only through the latent path: q keeps its gradient
    # (q_lat = q_flat @ w_q_lat), k does not (k_mean is detached - the exact
    # tiers and block means are constants w.r.t. the trainable params)
    assert q.grad is not None and q.grad.abs().sum() > 0
    assert k.grad is None
    # exact tier values unchanged: grad-mode output equals no-grad output
    out2, _ = three_tier_attention_fast(
        q.detach(),
        k.detach(),
        v.detach(),
        window,
        block,
        w_q_lat=w_q_lat.detach(),
        w_k_lat=w_k_lat.detach(),
    )
    torch.testing.assert_close(out.detach(), out2)
