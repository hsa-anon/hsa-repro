import logging

import torch

logger = logging.getLogger(__name__)


class TierLayerCache:
    # Streaming three-tier state for one attention layer, mirroring
    # three_tier_attention_fast decode-row semantics exactly:
    # window slice [idx-window+1, idx] inclusive of the query token,
    # landmark = block first token, visibility = block_last < window_start,
    # latent logit per completed block shared across heads (q = all-head
    # concat @ w_q_lat, k = block-mean all-kv-head concat @ w_k_lat,
    # latent value = block-mean v per kv head, GQA-expanded).
    # Eval is batch=1 only.
    def __init__(self, window, block_size):
        self.window = window
        self.block_size = block_size
        self.k = None
        self.v = None
        self.total = 0

    def seed(self, k, v):
        # k/v (T, Hkv, Dv) post-RoPE full-sequence prefill
        self.k = k
        self.v = v
        self.total = k.shape[0]

    def decode(self, q, k_new, v_new, w_q_lat, w_k_lat, latent_clamp=None, topk=None):
        # topk: query-dependent retrieval mode (LSA decode simulation) -
        # attend over window + exact top-k history keys by q.k score,
        # replacing landmark/latent tiers. Validates architecture pivot A:
        # retrieval must be query-dependent to reach arbitrary-depth keys.
        # q (Hq, D), k_new/v_new (Hkv, Dv). Appends the query token's own
        # k/v first (kernel window includes position idx), then attends
        # over [window | landmarks | latent]. Returns (Hq, Dv).
        self.k = torch.cat([self.k, k_new.unsqueeze(0)], dim=0)
        self.v = torch.cat([self.v, v_new.unsqueeze(0)], dim=0)
        self.total += 1
        T = self.total
        idx = T - 1
        ws = max(idx - self.window + 1, 0)
        kw = self.k[ws:T]  # (W, Hkv, D)
        vw = self.v[ws:T]  # (W, Hkv, Dv)
        n_blocks = (T + self.block_size - 1) // self.block_size
        bs = self.block_size
        Hkv, Dv = v_new.shape
        group = q.shape[0] // Hkv
        D = q.shape[-1]
        s_d = D**0.5
        L = w_k_lat.shape[-1] if w_k_lat is not None else 0
        s_l = L**0.5 if L else 1.0
        q_g = q.view(Hkv, group, D)

        if topk is not None:
            # exact retrieval (LSA decode simulation): per-kv-head-group
            # scores over the full history, top-k per head group,
            # softmax over retrieved keys only. einsum throughout: torch
            # 2.13 MPS matmul mis-broadcasts (Hkv,group,1,D)@(Hkv,group,D,K).
            kx = self.k[:T].permute(1, 0, 2)  # (Hkv, T, D)
            vx = self.v[:T].permute(1, 0, 2)  # (Hkv, T, Dv)
            scores = torch.einsum("hgd,htd->hgt", q_g, kx) / s_d  # (Hkv, group, T)
            K = min(topk, T)
            tk = scores.topk(K, dim=-1).indices  # (Hkv, group, K)
            sel_k = (
                kx.unsqueeze(1)
                .expand(Hkv, group, T, D)
                .gather(2, tk.unsqueeze(-1).expand(-1, -1, -1, D))
            )  # (Hkv, group, K, D)
            sel_v = (
                vx.unsqueeze(1)
                .expand(Hkv, group, T, Dv)
                .gather(2, tk.unsqueeze(-1).expand(-1, -1, -1, Dv))
            )  # (Hkv, group, K, Dv)
            probs = torch.softmax(torch.einsum("hgd,hgkd->hgk", q_g, sel_k) / s_d, dim=-1)
            out = torch.einsum("hgk,hgkd->hgd", probs, sel_v)  # (Hkv, group, Dv)
            return out.reshape(q.shape[0], Dv).to(q.dtype)

        logits = []
        w_len = T - ws
        logits.append(torch.matmul(q_g, kw.permute(1, 2, 0)) / s_d)  # (Hkv, group, W)
        visible = []
        lm_k = []
        lm_v = []
        for b in range(n_blocks):
            last = min((b + 1) * bs, T) - 1
            if last < ws:
                visible.append(b)
        if visible:
            lm_pos = torch.tensor([b * bs for b in visible], device=q.device)
            lm_k = self.k[lm_pos]  # (V, Hkv, D)
            lm_v = self.v[lm_pos]  # (V, Hkv, Dv)
            sl = torch.einsum("hgd,vhd->hgv", q_g, lm_k) / s_d  # (Hkv, group, V)
            logits.append(sl)
        if visible and L:
            k_mean = torch.stack(
                [self.k[b * bs : min((b + 1) * bs, T)].mean(dim=0) for b in visible], dim=0
            )
            v_mean = torch.stack(
                [self.v[b * bs : min((b + 1) * bs, T)].mean(dim=0) for b in visible], dim=0
            )
            k_lat = k_mean.reshape(len(visible), Hkv * Dv) @ w_k_lat  # (V, L)
            q_lat = q.reshape(-1) @ w_q_lat  # (L,)
            slt = (k_lat @ q_lat) / s_l  # (V,)
            if latent_clamp is not None:
                slt = slt.clamp(-latent_clamp, latent_clamp)
            logits.append(slt[None, None, :].expand(Hkv, group, -1))
        cat = torch.cat([x.reshape(Hkv * group, -1) for x in logits], dim=-1)  # (Hq, W+V+V)
        probs = torch.softmax(cat, dim=-1)
        with torch.no_grad():
            self.last_mass = (
                float(probs[:, :w_len].mean()),
                float(probs[:, w_len : w_len + len(visible)].mean()) if visible else 0.0,
                float(probs[:, w_len + len(visible) :].mean()) if visible and L else 0.0,
                float(cat.max()),
            )
        seg = 0
        out = torch.zeros(Hq := q.shape[0], Dv, device=q.device, dtype=torch.float32)
        pw = probs[:, seg : seg + w_len].view(Hkv, group, w_len)
        out = out + torch.einsum("hgw,hwd->hgd", pw, vw.permute(1, 0, 2)).reshape(Hq, Dv)
        seg += w_len
        if visible:
            Vn = len(visible)
            plm = probs[:, seg : seg + Vn].view(Hkv, group, Vn)
            out = out + torch.einsum("hgv,vhd->hgd", plm, lm_v).reshape(Hq, Dv)
            seg += Vn
            if L:
                plt_ = probs[:, seg : seg + Vn].view(Hkv, group, Vn)
                out = out + torch.einsum("hgv,vhd->hgd", plt_, v_mean).reshape(Hq, Dv)
        return out.to(q.dtype)
