import torch

from hsa.torch_tier_fast import three_tier_attention_fast, three_tier_attention_train


def _mk(B, T, Hq, Hkv, D, L, window, block):
    torch.manual_seed(11)
    q = torch.randn(B, T, Hq, D)
    k = torch.randn(B, T, Hkv, D)
    v = torch.randn(B, T, Hkv, D)
    wq = torch.randn(Hq * D, L) * 0.01
    wk = torch.randn(Hkv * D, L) * 0.01
    return q, k, v, wq, wk


def test_train_fn_matches_fast_forward():
    q, k, v, wq, wk = _mk(1, 16, 4, 2, 8, 3, window=4, block=8)
    ref, _ = three_tier_attention_fast(q, k, v, 4, 8, w_q_lat=wq, w_k_lat=wk)
    out, _ = three_tier_attention_train(q, k, v, 4, 8, wq, wk)
    torch.testing.assert_close(out, ref)


def test_train_fn_grads_match_reference():
    # The custom Function's w_q_lat/w_k_lat gradients must equal the
    # graph-mode path's gradients - the analytic backward replicates the
    # softmax row gradient restricted to the latent entries.
    q, k, v, wq, wk = _mk(1, 16, 4, 2, 8, 3, window=4, block=8)

    wq_ref = wq.clone().requires_grad_(True)
    wk_ref = wk.clone().requires_grad_(True)
    ref, _ = three_tier_attention_fast(q, k, v, 4, 8, w_q_lat=wq_ref, w_k_lat=wk_ref)
    ref.sum().backward()

    wq_tr = wq.clone().requires_grad_(True)
    wk_tr = wk.clone().requires_grad_(True)
    out, _ = three_tier_attention_train(q, k, v, 4, 8, wq_tr, wk_tr)
    out.sum().backward()

    torch.testing.assert_close(out.detach(), ref.detach())
    assert wq_ref.grad is not None and wk_ref.grad is not None
    torch.testing.assert_close(wq_tr.grad, wq_ref.grad, rtol=5e-2, atol=1e-3)
    torch.testing.assert_close(wk_tr.grad, wk_ref.grad, rtol=5e-2, atol=1e-3)


def test_train_fn_weighted_grads_match_reference():
    # Non-uniform upstream gradient: exercises the <dOut, v_Lb> term.
    q, k, v, wq, wk = _mk(1, 16, 4, 2, 8, 3, window=4, block=8)
    g = torch.randn(1, 16, 4, 8)

    wq_ref = wq.clone().requires_grad_(True)
    wk_ref = wk.clone().requires_grad_(True)
    ref, _ = three_tier_attention_fast(q, k, v, 4, 8, w_q_lat=wq_ref, w_k_lat=wk_ref)
    (ref * g).sum().backward()

    wq_tr = wq.clone().requires_grad_(True)
    wk_tr = wk.clone().requires_grad_(True)
    out, _ = three_tier_attention_train(q, k, v, 4, 8, wq_tr, wk_tr)
    (out * g).sum().backward()

    torch.testing.assert_close(wq_tr.grad, wq_ref.grad, rtol=5e-2, atol=1e-3)
    torch.testing.assert_close(wk_tr.grad, wk_ref.grad, rtol=5e-2, atol=1e-3)


def test_train_fn_qkv_grads_are_none_but_chain_flows():
    # q/k/v grads are intentionally None (cut path, no trainable parameter
    # behind them); the Function must still be differentiable end-to-end
    # for the weights.
    q, k, v, wq, wk = _mk(1, 16, 4, 2, 8, 3, window=4, block=8)
    wq_tr = wq.clone().requires_grad_(True)
    wk_tr = wk.clone().requires_grad_(True)
    out, _ = three_tier_attention_train(q, k, v, 4, 8, wq_tr, wk_tr)
    out.sum().backward()
    assert wq_tr.grad is not None
    assert torch.isfinite(wq_tr.grad).all()
    assert torch.isfinite(wk_tr.grad).all()
