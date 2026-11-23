import logging

import torch

logger = logging.getLogger(__name__)


class NsaLayerCache:
    # Decode-side port of Native Sparse Attention (Yuan et al. 2025,
    # DeepSeek): three-branch decode attention with (1) a compression branch
    # scoring block-mean keys, (2) a top-N fine block selection driven by the
    # coarse scores, and (3) a sliding window that always attends. The paper
    # trains intra-block compression MLPs; this port uses the untrained
    # block mean, which is the paper's structure with identity compression -
    # the same construction as our latent tier. Hardware kernels (FlashMLA
    # paths) are not ported; eval is batch=1. Fine budget beyond the window
    # is n_blocks * block_size keys, the matched-budget comparison unit
    # against the exact topk retrieval path.
    def __init__(self, block_size=32, n_blocks=2, window=512):
        self.block_size = block_size
        self.n_blocks = n_blocks
        self.window = window
        self.k = None
        self.v = None
        self.total = 0

    def seed(self, k, v):
        self.k = k
        self.v = v
        self.total = k.shape[0]

    def append_chunk(self, k, v):
        self.k = torch.cat([self.k, k], dim=0)
        self.v = torch.cat([self.v, v], dim=0)
        self.total += k.shape[0]

    def _append(self, k_new, v_new):
        self.k = torch.cat([self.k, k_new.unsqueeze(0)], dim=0)
        self.v = torch.cat([self.v, v_new.unsqueeze(0)], dim=0)
        self.total += 1

    def decode(self, q, k_new, v_new, n_blocks=None):
        # q (Hq, D) post-RoPE, k_new/v_new (Hkv, Dv). Appends the token, then
        # per GQA group: compressed coarse scores -> top-N blocks (current
        # block always attended, excluded from competition); softmax over the
        # union of fine tokens (selected blocks + window) and the compressed
        # keys. Returns (Hq, Dv).
        self._append(k_new, v_new)
        T = self.total
        n = self.n_blocks if n_blocks is None else n_blocks
        bs = self.block_size
        Hkv, Dv = v_new.shape
        Hq, D = q.shape
        s_d = D**0.5
        q_g = q.reshape(Hkv, Hq // Hkv, D).to(torch.float32)
        kx = self.k[:T].permute(1, 0, 2).to(torch.float32)  # (Hkv, T, D)
        vx = self.v[:T].permute(1, 0, 2).to(torch.float32)  # (Hkv, T, Dv)
        n_blk = (T + bs - 1) // bs
        pad = n_blk * bs - T
        if pad:
            kp = torch.cat([kx, kx.new_zeros(Hkv, pad, D)], dim=1)
            vp = torch.cat([vx, vx.new_zeros(Hkv, pad, Dv)], dim=1)
        else:
            kp, vp = kx, vx
        kp = kp.reshape(Hkv, n_blk, bs, D)
        valid = torch.zeros(n_blk, bs, device=kx.device, dtype=torch.bool)
        valid.view(-1)[:T] = True
        # Compression branch: untrained block-mean keys (see class docstring).
        cmp_k = (kp * valid.unsqueeze(-1)).sum(dim=2) / valid.sum(dim=1, keepdim=True).clamp(min=1)
        cmp_scores = torch.einsum("hgd,hbd->hgb", q_g, cmp_k) / s_d  # (Hkv, group, n_blk)
        cur_blk = (T - 1) // bs
        cmp_scores = cmp_scores.clone()
        cmp_scores[:, :, cur_blk] = float("-inf")  # fine branch covers it
        if pad:
            cmp_scores[:, :, -1] = float("-inf")  # padded block never selected
        n_sel = min(n, n_blk - 1)
        blk_of = torch.arange(T, device=q.device) // bs
        if n_sel > 0:
            sel = cmp_scores.topk(n_sel, dim=-1).indices  # (Hkv, group, n_sel)
            in_sel = (blk_of.view(1, 1, T) == sel.unsqueeze(-1)).any(dim=2)
        else:
            in_sel = torch.zeros(1, 1, T, dtype=torch.bool, device=q.device)
        in_win = (torch.arange(T, device=q.device) >= T - self.window) | (blk_of == cur_blk)
        token_mask = in_sel | in_win.view(1, 1, T)
        # Joint softmax over fine tokens + compressed keys (the paper's three
        # branch outputs accumulate into one normalized attention; at eval
        # batch=1 this is a single softmax over the union of the branches).
        scores = torch.einsum("hgd,htd->hgt", q_g, kx) / s_d  # (Hkv, group, T)
        scores = scores.masked_fill(~token_mask, float("-inf"))
        probs = torch.softmax(torch.cat([scores, cmp_scores], dim=-1), dim=-1)
        fine_out = torch.einsum("hgt,htd->hgd", probs[..., :T], vx)
        cmp_out = torch.einsum("hgb,hbd->hgd", probs[..., T:], cmp_k)
        out = fine_out + cmp_out
        return out.reshape(Hq, Dv).to(q.dtype)
