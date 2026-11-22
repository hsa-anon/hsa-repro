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


def _manual_slt(q, k, wq, wk, window, block):
    # Visible-subset softmax over the latent block logits (same construction
    # as the kernel's aux path): masked exp normalized, no fmin fill.
    B, T, Hq, D = q.shape
    Hkv = k.shape[2]
    nb = (T + block - 1) // block
    counts = torch.tensor(
        [min((b + 1) * block, T) - b * block for b in range(nb)], dtype=torch.float32
    )
    k_mean = k.reshape(B, nb, block, Hkv, D).float().sum(2) / counts[None, :, None, None]
    q_lat = (q.reshape(B, T, Hq * D) @ wq).float()
    k_lat = (k_mean.reshape(B, nb, Hkv * D) @ wk).float()
    slt = torch.einsum("btl,bnl->btn", q_lat, k_lat) / (_L := wq.shape[-1]) ** 0.5
    idx = torch.arange(T)
    ws = (idx - window + 1).clamp(min=0)
    block_last = torch.tensor([min((b + 1) * block, T) - 1 for b in range(nb)])
    vis = block_last[None, :] < ws[:, None]
    neg = torch.finfo(torch.float32).min
    slt = slt.masked_fill(~vis[None], neg)
    mx = slt.max(-1, keepdim=True).values
    exps = (slt - mx).exp() * vis[None]
    p = exps / exps.sum(-1, keepdim=True).clamp_min(1e-8)
    return p, vis


def test_block_target_grad_matches_reference():
    # Aux CE on the latent block distribution at sampled rows: the analytic
    # backward with a block target must equal graph-mode fast forward +
    # manual CE on the recomputed latent block softmax (all rows sampled).
    # Geometry needs rows with >= 2 visible blocks: a single-visible-block
    # row's subset softmax is one-hot, so p == tgt == one-hot and the aux
    # grad is legitimately zero there.
    q, k, v, wq, wk = _mk(1, 32, 4, 2, 8, 3, window=4, block=8)
    T, nb, Hq = 32, 4, 4
    _p_ref, vis = _manual_slt(q, k, wq, wk, 4, 8)
    assert (vis.sum(-1) >= 2).any()
    # Non-one-hot targets on the visible support, zero mass elsewhere (same
    # normalization block_mass.py produces); invisible rows get target 0.
    torch.manual_seed(3)
    soft = torch.softmax(torch.randn(T, nb) * 0.1, dim=-1)
    target = soft * vis
    target = target / target.sum(-1, keepdim=True).clamp_min(1e-8)
    lam = 2.0
    rows = torch.arange(T)

    wq_a = wq.clone().requires_grad_(True)
    wk_a = wk.clone().requires_grad_(True)
    out, _ = three_tier_attention_train(
        q, k, v, 4, 8, wq_a, wk_a, block_target=(rows, target), block_weight=lam
    )
    out.sum().backward()

    wq_r = wq.clone().requires_grad_(True)
    wk_r = wk.clone().requires_grad_(True)
    ref, _ = three_tier_attention_fast(q, k, v, 4, 8, w_q_lat=wq_r, w_k_lat=wk_r)
    p_r, _ = _manual_slt(q, k, wq_r, wk_r, 4, 8)
    ce = -(target * p_r.clamp_min(1e-30).log()).sum()
    (ref.sum() + lam * Hq * ce).backward()

    torch.testing.assert_close(wq_a.grad, wq_r.grad, rtol=5e-2, atol=1e-4)
    torch.testing.assert_close(wk_a.grad, wk_r.grad, rtol=5e-2, atol=1e-4)


def test_block_target_rows_sparse():
    # Gradient lands only at the sampled rows; unsampled rows keep the
    # main-loss-only gradient.
    q, k, v, wq, wk = _mk(1, 32, 4, 2, 8, 3, window=4, block=8)
    T, nb, Hq = 32, 4, 4
    _p_ref, vis = _manual_slt(q, k, wq, wk, 4, 8)
    torch.manual_seed(3)
    soft = torch.softmax(torch.randn(T, nb) * 0.1, dim=-1)
    target = soft * vis
    target = target / target.sum(-1, keepdim=True).clamp_min(1e-8)
    rows = torch.tensor([7, 19, 22, 31])

    wq_all = wq.clone().requires_grad_(True)
    out, _ = three_tier_attention_train(
        q,
        k,
        v,
        4,
        8,
        wq_all,
        wk.clone().requires_grad_(True),
        block_target=(torch.arange(T), target),
    )
    out.sum().backward()

    wq_sp = wq.clone().requires_grad_(True)
    out2, _ = three_tier_attention_train(
        q, k, v, 4, 8, wq_sp, wk.clone().requires_grad_(True), block_target=(rows, target[rows])
    )
    out2.sum().backward()

    # sparse rows subset: dense run = sparse run + extra-row contributions.
    # Sanity: both finite, sparse grad differs from dense (rows dropped), and
    # equals a manual recomputation with only those rows in the CE.
    assert torch.isfinite(wq_sp.grad).all()
    assert not torch.allclose(wq_all.grad, wq_sp.grad)
    wq_m = wq.clone().requires_grad_(True)
    ref, _ = three_tier_attention_fast(
        q, k, v, 4, 8, w_q_lat=wq_m, w_k_lat=wk.clone().requires_grad_(True)
    )
    p_m, _ = _manual_slt(q, k, wq_m, wk, 4, 8)
    ce = -(target[rows] * p_m[:, rows].clamp_min(1e-30).log()).sum()
    (ref.sum() + Hq * ce).backward()
    torch.testing.assert_close(wq_sp.grad, wq_m.grad, rtol=5e-2, atol=1e-4)


def test_block_target_large_logits_finite():
    # Trained latents reach ~660 magnitude: raw exp overflows fp32 and the
    # aux path produced NaN student losses on the NPU (2026-11-15). The
    # max-shifted subset softmax must stay finite.
    q, k, v, wq, wk = _mk(1, 32, 4, 2, 8, 3, window=4, block=8)
    T, nb = 32, 4
    wq = wq * 20000.0
    wk = wk * 20000.0
    _, vis = _manual_slt(q, k, wq, wk, 4, 8)
    torch.manual_seed(3)
    soft = torch.softmax(torch.randn(T, nb) * 0.1, dim=-1)
    target = soft * vis
    target = target / target.sum(-1, keepdim=True).clamp_min(1e-8)
    rows = torch.arange(T)

    wq_a = wq.clone().requires_grad_(True)
    out, _ = three_tier_attention_train(
        q, k, v, 4, 8, wq_a, wk.clone().requires_grad_(True), block_target=(rows, target)
    )
    out.sum().backward()
    assert torch.isfinite(out).all()
    assert torch.isfinite(wq_a.grad).all()


def test_block_target_none_unchanged():
    # No target: identical to the pre-existing path (main grads only).
    q, k, v, wq, wk = _mk(1, 16, 4, 2, 8, 3, window=4, block=8)
    wq_a = wq.clone().requires_grad_(True)
    wk_a = wk.clone().requires_grad_(True)
    out, _ = three_tier_attention_train(q, k, v, 4, 8, wq_a, wk_a, block_target=None)
    out.sum().backward()
    wq_b = wq.clone().requires_grad_(True)
    wk_b = wk.clone().requires_grad_(True)
    out2, _ = three_tier_attention_train(q, k, v, 4, 8, wq_b, wk_b)
    out2.sum().backward()
    torch.testing.assert_close(wq_a.grad, wq_b.grad)
    torch.testing.assert_close(wk_a.grad, wk_b.grad)
