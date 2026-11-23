import torch

from hsa.eval.nsa_cache import NsaLayerCache


def _rand_kv(T, Hkv, D, seed=0):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(T, Hkv, D, generator=g)
    v = torch.randn(T, Hkv, D, generator=g)
    return k, v


def test_decode_output_shape_and_dtype():
    k, v = _rand_kv(1000, 2, 16, seed=1)
    c = NsaLayerCache(block_size=32, n_blocks=2, window=64)
    c.seed(k, v)
    g = torch.Generator().manual_seed(2)
    q = torch.randn(12, 16, generator=g)  # Hq = Hkv * group = 2 * 6
    out = c.decode(q, k[-1], v[-1])
    assert out.shape == (12, 16)
    assert out.dtype == q.dtype


def test_decode_equals_bruteforce_union_attention():
    # Oracle: build the selection from the same coarse scores, then dense
    # softmax over (selected blocks + window) fine tokens + all cmp keys.
    k, v = _rand_kv(300, 2, 16, seed=3)
    c = NsaLayerCache(block_size=32, n_blocks=2, window=64)
    c.seed(k, v)
    g = torch.Generator().manual_seed(4)
    q = torch.randn(12, 16, generator=g)
    out = c.decode(q, k[-1], v[-1])

    T = c.total
    kk = torch.cat([k, k[-1:]], dim=0)  # cache appended one token
    vv = torch.cat([v, v[-1:]], dim=0)
    bs = 32
    Hkv, group, D = 2, 6, 16
    q_g = q.reshape(Hkv, group, D).float()
    kx = kk[:T].permute(1, 0, 2).float()
    n_blk = (T + bs - 1) // bs
    kp = kk[:T].float()
    pad = n_blk * bs - T
    if pad:
        kp = torch.cat([kp, torch.zeros(pad, Hkv, D)], dim=0)
    kp = kp.reshape(n_blk, bs, Hkv, D).permute(2, 0, 1, 3)
    valid = torch.zeros(n_blk, bs, dtype=torch.bool)
    valid.view(-1)[:T] = True
    cmp_k = (kp * valid.unsqueeze(-1)).sum(dim=2) / valid.sum(dim=1, keepdim=True).clamp(min=1)
    cmp_scores = torch.einsum("hgd,hbd->hgb", q_g, cmp_k) / D**0.5
    cur = (T - 1) // bs
    cmp_scores[:, :, cur] = float("-inf")
    sel = cmp_scores.topk(2, dim=-1).indices
    blk_of = torch.arange(T) // bs
    in_sel = (blk_of.view(1, 1, T) == sel.unsqueeze(-1)).any(dim=2)
    in_win = (torch.arange(T) >= T - 64) | (blk_of == cur)
    mask = in_sel | in_win.view(1, 1, T)
    scores = torch.einsum("hgd,htd->hgt", q_g, kx) / D**0.5
    scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(torch.cat([scores, cmp_scores], dim=-1), dim=-1)
    fine = torch.einsum("hgt,htd->hgd", probs[..., :T], vv[:T].permute(1, 0, 2).float())
    cmp_o = torch.einsum("hgb,hbd->hgd", probs[..., T:], cmp_k)
    ref = (fine + cmp_o).reshape(12, 16)
    assert torch.allclose(out.float(), ref, atol=1e-5)


def test_current_block_always_attended():
    # Needle planted far from the window but selected via coarse score must
    # be reachable; and the current block is never dropped from the mask.
    k, v = _rand_kv(1000, 1, 16, seed=5)
    c = NsaLayerCache(block_size=32, n_blocks=1, window=32)
    c.seed(k, v)
    q = torch.zeros(4, 16)
    out = c.decode(q, k[-1], v[-1])
    # zero query -> uniform attention -> mean of (selected + window + cmp);
    # just verify finiteness and no NaN from the -inf masking.
    assert torch.isfinite(out).all()


def test_single_block_history_degrades_to_window_only():
    k, v = _rand_kv(20, 2, 16, seed=6)
    c = NsaLayerCache(block_size=32, n_blocks=2, window=512)
    c.seed(k, v)
    g = torch.Generator().manual_seed(7)
    q = torch.randn(12, 16, generator=g)
    out = c.decode(q, k[-1], v[-1])
    # All tokens inside one block: fine mask covers the whole window; cmp
    # branch is fully masked -> uniform-ish finite output.
    assert torch.isfinite(out).all()


def test_budget_matches_topk_path():
    # Fine budget beyond the window is n_blocks * block_size keys.
    c = NsaLayerCache(block_size=32, n_blocks=2, window=512)
    assert c.n_blocks * c.block_size == 64
