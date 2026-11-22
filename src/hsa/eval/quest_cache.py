import logging

import torch

logger = logging.getLogger(__name__)


class QuestLayerCache:
    # Official Quest semantics (mit-han-lab/quest, Tang et al. 2024), ported
    # line-by-line from the CUDA kernels to torch ops: dense KV storage per
    # full-attention layer, per-page per-channel key min/max maintained
    # incrementally, decode scores every candidate page with the box upper
    # bound sum_d max(q_d * kmax[h,p,d], q_d * kmin[h,p,d]) (no ReLU, no
    # clamp - decode_attn.cuh compute_max_possible), selects the top
    # (`pages` - 1) pages per query head excluding the current page, and
    # always attends over the current page on top of the selection
    # (decode_estimate excludes the last metadata entry; the sparse decode
    # kernel re-adds the last page via last_page_idx). No window guarantee
    # and no learned tiers: selection is purely query-dependent.
    # Eval is batch=1 only. Page granularity is the comparison unit against
    # the exact topk retrieval path (pages * page_size tokens vs topk keys).
    def __init__(self, page_size=64):
        self.page_size = page_size
        self.k = None
        self.v = None
        self.kmin = None
        self.kmax = None
        self.total = 0

    def seed(self, k, v):
        # k/v (T, Hkv, Dv) post-RoPE full-sequence prefill
        self.k = k
        self.v = v
        self.total = k.shape[0]
        n_pages = (self.total + self.page_size - 1) // self.page_size
        self.kmin, self.kmax = self._page_bounds(k, n_pages)

    def _page_bounds(self, k, n_pages):
        T, Hkv, D = k.shape
        need = n_pages * self.page_size
        kf = k.to(torch.float32)
        if need != T:
            kf = torch.cat([kf, torch.zeros(need - T, Hkv, D, device=k.device)], dim=0)
            valid = torch.zeros(need, device=k.device, dtype=torch.bool)
            valid[:T] = True
        else:
            valid = torch.ones(need, device=k.device, dtype=torch.bool)
        p = kf.reshape(n_pages, self.page_size, Hkv, D)
        m = valid.reshape(n_pages, self.page_size, 1, 1)
        kmin = p.masked_fill(~m, float("inf")).min(dim=1).values
        kmax = p.masked_fill(~m, float("-inf")).max(dim=1).values
        return kmin, kmax

    def append_chunk(self, k, v):
        # Chunked-prefill append: k/v (t, Hkv, Dv). Concatenate and refresh
        # bounds for every page the chunk touches (recompute from raw keys -
        # O(chunk + page) per chunk, negligible against the dense forward).
        old = self.total
        self.k = torch.cat([self.k, k], dim=0)
        self.v = torch.cat([self.v, v], dim=0)
        self.total += k.shape[0]
        n_pages = (self.total + self.page_size - 1) // self.page_size
        if n_pages > self.kmin.shape[0]:
            pad = n_pages - self.kmin.shape[0]
            self.kmin = torch.cat(
                [
                    self.kmin,
                    torch.zeros(
                        pad, self.kmin.shape[1], self.kmin.shape[2], device=self.kmin.device
                    ),
                ],
                dim=0,
            )
            self.kmax = torch.cat(
                [
                    self.kmax,
                    torch.zeros(
                        pad, self.kmax.shape[1], self.kmax.shape[2], device=self.kmax.device
                    ),
                ],
                dim=0,
            )
        first_page = old // self.page_size
        start = first_page * self.page_size
        n_new = (self.total - start + self.page_size - 1) // self.page_size
        kmin, kmax = self._page_bounds(self.k[start : self.total], n_new)
        self.kmin[first_page : first_page + n_new] = kmin
        self.kmax[first_page : first_page + n_new] = kmax

    def _append(self, k_new, v_new):
        # k_new/v_new (Hkv, Dv): append one decode token, refresh the min/max
        # of its page (open a new page when crossing the boundary).
        self.k = torch.cat([self.k, k_new.unsqueeze(0)], dim=0)
        self.v = torch.cat([self.v, v_new.unsqueeze(0)], dim=0)
        self.total += 1
        T = self.total
        n_pages = (T + self.page_size - 1) // self.page_size
        if n_pages > self.kmin.shape[0]:
            self.kmin = torch.cat([self.kmin, self.k[T - 1].unsqueeze(0).to(torch.float32)], dim=0)
            self.kmax = torch.cat([self.kmax, self.k[T - 1].unsqueeze(0).to(torch.float32)], dim=0)
        else:
            start = ((T - 1) // self.page_size) * self.page_size
            seg = self.k[start:T].to(torch.float32)
            self.kmin[(T - 1) // self.page_size] = seg.min(dim=0).values
            self.kmax[(T - 1) // self.page_size] = seg.max(dim=0).values

    def decode(self, q, k_new, v_new, pages):
        # q (Hq, D) post-RoPE, k_new/v_new (Hkv, Dv). Appends the token, then
        # scores every candidate page per query head with the official Quest
        # box upper bound:
        #   bound[h,g,p] = sum_d max(q_d * kmax[h,p,d], q_d * kmin[h,p,d])
        # selects the top-(`pages` - 1) candidate pages per (kv head, query
        # group) with the current page excluded from competition, and
        # softmax-attends over the union of selected pages plus the current
        # page. Returns (Hq, Dv).
        self._append(k_new, v_new)
        T = self.total
        n_pages = (T + self.page_size - 1) // self.page_size
        Hkv, Dv = v_new.shape
        Hq, D = q.shape
        group = Hq // Hkv
        s_d = D**0.5
        q_g = q.reshape(Hkv, group, D).to(torch.float32)
        kmin = self.kmin.permute(1, 0, 2)  # (Hkv, n_pages, D)
        kmax = self.kmax.permute(1, 0, 2)
        prod_a = q_g.unsqueeze(2) * kmin.unsqueeze(1)  # (Hkv, group, n_pages, D)
        prod_b = q_g.unsqueeze(2) * kmax.unsqueeze(1)
        bound = torch.maximum(prod_a, prod_b).sum(dim=-1)  # (Hkv, group, n_pages)
        cur_page = (T - 1) // self.page_size
        n_sel = min(pages - 1, n_pages - 1)
        bound = bound.clone()
        bound[:, :, cur_page] = float("-inf")  # current page never competes
        sel = bound.topk(n_sel, dim=-1).indices  # (Hkv, group, n_sel)
        kx = self.k[:T].permute(1, 0, 2).to(torch.float32)  # (Hkv, T, D)
        vx = self.v[:T].permute(1, 0, 2).to(torch.float32)
        page_of = torch.arange(T, device=q.device) // self.page_size  # (T,)
        in_sel = (page_of.view(1, 1, T) == sel.unsqueeze(-1)).any(dim=2)  # (Hkv, group, T)
        in_sel = in_sel | (page_of == cur_page).view(1, 1, T)
        scores = torch.einsum("hgd,htd->hgt", q_g, kx) / s_d
        scores = scores.masked_fill(~in_sel, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("hgt,htd->hgd", probs, vx)  # (Hkv, group, Dv)
        return out.reshape(Hq, Dv).to(q.dtype)
