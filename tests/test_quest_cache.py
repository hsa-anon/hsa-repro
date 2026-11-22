import math

import torch

from hsa.eval.quest_cache import QuestLayerCache


def _rand_kv(T, Hkv, D, seed=0):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(T, Hkv, D, generator=g)
    v = torch.randn(T, Hkv, D, generator=g)
    return k, v


def test_page_bounds_match_full_recompute():
    k, _ = _rand_kv(300, 2, 16, seed=1)
    c = QuestLayerCache(page_size=64)
    c.seed(k, torch.zeros_like(k))
    n = (300 + 63) // 64
    kmin_ref, kmax_ref = c._page_bounds(k, n)
    assert torch.allclose(c.kmin, kmin_ref)
    assert torch.allclose(c.kmax, kmax_ref)


def test_append_chunk_bounds_match_seed():
    k, v = _rand_kv(1000, 2, 16, seed=2)
    whole = QuestLayerCache(page_size=64)
    whole.seed(k, v)
    chunked = QuestLayerCache(page_size=64)
    chunked.seed(k[:100], v[:100])
    for s in range(100, 1000, 137):
        e = min(s + 137, 1000)
        chunked.append_chunk(k[s:e], v[s:e])
    assert chunked.total == whole.total
    assert torch.allclose(chunked.kmin, whole.kmin)
    assert torch.allclose(chunked.kmax, whole.kmax)


def test_decode_token_bounds_match_seed():
    k, v = _rand_kv(200, 2, 16, seed=3)
    c = QuestLayerCache(page_size=64)
    c.seed(k, v)
    g2 = torch.Generator().manual_seed(4)
    kn = torch.randn(2, 16, generator=g2)
    vn = torch.randn(2, 16, generator=g2)
    c._append(kn, vn)
    ref = QuestLayerCache(page_size=64)
    ref.seed(torch.cat([k, kn.unsqueeze(0)]), torch.cat([v, vn.unsqueeze(0)]))
    assert torch.allclose(c.kmin, ref.kmin)
    assert torch.allclose(c.kmax, ref.kmax)
    assert c.total == ref.total


def test_bound_is_valid_upper_estimate_for_aligned_pages():
    # A page whose keys are strongly aligned with q (positive channels, large
    # magnitude) must receive a bound at least as large as any flat page's.
    T, Hkv, D, P = 256, 1, 16, 64
    k, v = _rand_kv(T, Hkv, D, seed=5)
    needle_page = 2
    k[needle_page * P : (needle_page + 1) * P] = 5.0
    c = QuestLayerCache(page_size=P)
    c.seed(k, v)
    q = torch.ones(Hkv * 1, D)
    c._append(torch.ones(Hkv, D), torch.zeros(Hkv, D))
    q_g = q.view(Hkv, 1, D)
    kmin = c.kmin.permute(1, 0, 2)
    kmax = c.kmax.permute(1, 0, 2)
    prod_a = q_g.unsqueeze(2) * kmin.unsqueeze(1)
    prod_b = q_g.unsqueeze(2) * kmax.unsqueeze(1)
    bound = torch.maximum(prod_a, prod_b).sum(dim=-1)
    assert bound[0, 0, needle_page] == bound[0, 0].max()


def test_decode_equals_exact_attention_over_selected_pages():
    T, Hkv, Hq, D, P, pages = 128, 2, 4, 8, 16, 2
    k, v = _rand_kv(T, Hkv, D, seed=6)
    c = QuestLayerCache(page_size=P)
    c.seed(k, v)
    g = torch.Generator().manual_seed(7)
    q = torch.randn(Hq, D, generator=g)
    kn = torch.randn(Hkv, D, generator=g)
    vn = torch.randn(Hkv, D, generator=g)
    out = c.decode(q, kn, vn, pages)

    k_all = torch.cat([k, kn.unsqueeze(0)])
    v_all = torch.cat([v, vn.unsqueeze(0)])
    T_all = T + 1
    n_pages = (T_all + P - 1) // P
    kmin, kmax = c._page_bounds(k_all, n_pages)
    q_g = q.view(Hkv, Hq // Hkv, D)
    prod_a = q_g.unsqueeze(2) * kmin.permute(1, 0, 2).unsqueeze(1)
    prod_b = q_g.unsqueeze(2) * kmax.permute(1, 0, 2).unsqueeze(1)
    bound = torch.maximum(prod_a, prod_b).sum(dim=-1)
    cur_page = (T_all - 1) // P
    bound = bound.clone()
    bound[:, :, cur_page] = float("-inf")
    sel = bound.topk(pages - 1, dim=-1).indices
    page_of = torch.arange(T_all) // P
    in_sel = (page_of.view(1, 1, T_all) == sel.unsqueeze(-1)).any(dim=2)
    in_sel = in_sel | (page_of == cur_page).view(1, 1, T_all)
    for h in range(Hkv):
        for gi in range(Hq // Hkv):
            idx = torch.nonzero(in_sel[h, gi]).squeeze(-1)
            ks = k_all[idx, h]
            vs = v_all[idx, h]
            qh = q[h * (Hq // Hkv) + gi]
            scores = ks @ qh / math.sqrt(D)
            probs = torch.softmax(scores, dim=-1)
            ref = probs @ vs
            assert torch.allclose(out[h * (Hq // Hkv) + gi], ref, atol=1e-5)


def test_pages_budget_larger_than_history_selects_all():
    T, Hkv, Hq, D = 30, 1, 2, 8
    k, v = _rand_kv(T, Hkv, D, seed=8)
    c = QuestLayerCache(page_size=64)
    c.seed(k, v)
    q = torch.randn(Hq, D)
    kn = torch.randn(Hkv, D)
    vn = torch.randn(Hkv, D)
    out = c.decode(q, kn, vn, pages=4)
    # pages >= n_pages: every token attends; decode == full dense attention
    k_all = torch.cat([k, kn.unsqueeze(0)])
    v_all = torch.cat([v, vn.unsqueeze(0)])
    for gi in range(Hq):
        scores = k_all[:, 0] @ q[gi] / math.sqrt(D)
        ref = torch.softmax(scores, dim=-1) @ v_all[:, 0]
        assert torch.allclose(out[gi], ref, atol=1e-5)


def test_decode_matches_official_quest_semantics_bruteforce():
    # Official Quest (decode_attn.cuh compute_max_possible + decode_wrapper):
    # bound = sum_d max(q_d * kmax_d, q_d * kmin_d); the current page is
    # excluded from topk competition and always attended.
    T, Hkv, Hq, D, P, pages = 300, 4, 8, 64, 64, 3
    k, v = _rand_kv(T, Hkv, D, seed=9)
    c = QuestLayerCache(page_size=P)
    c.seed(k[:200], v[:200])
    c.append_chunk(k[200:250], v[200:250])
    g = torch.Generator().manual_seed(10)
    q = torch.randn(Hq, D, generator=g)
    out = c.decode(q, k[250], v[250], pages=pages)

    n_pages = (251 + P - 1) // P
    pad = n_pages * P - 251
    k_all = torch.cat([k[:251].float(), torch.zeros(pad, Hkv, D)], dim=0)
    v_all = v[:251].float()
    page_of = torch.arange(251) // P
    cur_page = page_of[-1]
    m = torch.zeros(n_pages * P, dtype=torch.bool)
    m[:251] = True
    kr = k_all.reshape(n_pages, P, Hkv, D)
    kmin = kr.masked_fill(~m.reshape(n_pages, P, 1, 1), float("inf")).amin(1).permute(1, 0, 2)
    kmax = kr.masked_fill(~m.reshape(n_pages, P, 1, 1), float("-inf")).amax(1).permute(1, 0, 2)
    q_g = q.reshape(Hkv, Hq // Hkv, D).float()
    bound = torch.maximum(q_g[:, :, None] * kmax[:, None], q_g[:, :, None] * kmin[:, None]).sum(-1)
    bound[:, :, cur_page] = float("-inf")
    sel = bound.topk(pages - 1, dim=-1).indices
    for h in range(Hkv):
        for gi in range(Hq // Hkv):
            keep = (page_of == cur_page) | torch.isin(page_of, sel[h, gi])
            s = (q_g[h, gi] * k_all[:251, h]).sum(-1) / math.sqrt(D)
            s = s.masked_fill(~keep, float("-inf"))
            ref = torch.softmax(s, -1) @ v_all[:, h]
            assert torch.allclose(out[h * (Hq // Hkv) + gi].float(), ref, atol=1e-4)


def test_decode_always_attends_current_page():
    # Even with a budget of one page, the page holding the query token must
    # be attended (official: estimate excludes it, sparse attn re-adds it).
    T, Hkv, Hq, D, P = 64, 1, 2, 8, 64
    k, v = _rand_kv(T, Hkv, D, seed=11)
    c = QuestLayerCache(page_size=P)
    c.seed(k, v)
    q = torch.zeros(Hq, D)
    kn = torch.zeros(Hkv, D)
    vn = torch.ones(Hkv, D) * 7.0
    out = c.decode(q, kn, vn, pages=1)
    # q=0 -> uniform scores over the single attended page -> mean of values
    assert torch.allclose(out, torch.full_like(out, 7.0), atol=1e-5)
