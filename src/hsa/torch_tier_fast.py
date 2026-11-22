import logging

import torch

logger = logging.getLogger(__name__)


def three_tier_attention_fast(
    q,
    k,
    v,
    window,
    block_size,
    w_q_lat=None,
    w_k_lat=None,
    latent_clamp=None,
    q_offset=0,
    latent_zscore=False,
    return_probs_lat=False,
):
    """Vectorized three-tier attention for training throughput.

    Shapes: q (B, Tq, Hq, D), k/v (B, Tabs, Hkv, Dv). Query rows are the
    absolute positions [q_offset, q_offset + Tq); chunked prefill passes
    the last chunk as q with q_offset = chunk start and the whole history
    as k/v. q_offset=0 with Tq == Tabs is the plain case. Exact window +
    exact landmark keys (block first tokens) + shared-latent gist:
    block-mean k over concatenated kv heads -> W_k_lat (Hkv*D -> L),
    token-concat query -> W_q_lat (Hq*D -> L), one latent logit per
    (batch, query, block) broadcast across heads. Latent values =
    block-mean v per kv head, GQA-expanded. Single softmax per head over
    [window | landmarks | latent]. w_q_lat/w_k_lat near-zero init keeps
    step-0 student = window + landmarks only.
    """
    B, T, Hq, D = q.shape
    Hkv = k.shape[2]
    Dv = v.shape[-1]
    Tabs = k.shape[1]
    group = Hq // Hkv
    L = w_k_lat.shape[-1] if w_k_lat is not None else 0
    device = q.device
    dtype = q.dtype
    s_d = D**0.5
    s_l = L**0.5 if L else 1.0

    n_blocks = (Tabs + block_size - 1) // block_size
    pad = n_blocks * block_size - Tabs
    if pad:
        k = torch.cat([k, torch.zeros(B, pad, Hkv, Dv, dtype=dtype, device=device)], dim=1)
        v = torch.cat([v, torch.zeros(B, pad, Hkv, Dv, dtype=dtype, device=device)], dim=1)

    # Everything below except the latent logits is a pure function of the
    # frozen backbone (window/landmark scores, value gathers, block means).
    # Computing it under no_grad drops the ~8.2GB window-gather graph at 8k
    # (twice: scores + values) and the backbone gradient flow through the
    # exact tiers - the only trainable tensors are w_q_lat/w_k_lat, which
    # sit entirely on the latent path. The cut cross-layer path (earlier
    # tier layers' latent outputs feeding later layers' exact scores) is a
    # second-order approximation, ledgered as a ruling.
    with torch.no_grad():
        k5 = k.reshape(B, n_blocks, block_size, Hkv, Dv)
        v5 = v.reshape(B, n_blocks, block_size, Hkv, Dv)
        counts = torch.tensor(
            [min((b + 1) * block_size, Tabs) - b * block_size for b in range(n_blocks)],
            dtype=dtype,
            device=device,
        )[None, :, None, None]
        k_mean = k5.sum(dim=2) / counts
        v_mean = v5.sum(dim=2) / counts
        block_last = torch.tensor(
            [min((b + 1) * block_size, Tabs) - 1 for b in range(n_blocks)], device=device
        )

        idx = q_offset + torch.arange(T, device=device)
        ws = torch.clamp(idx - window + 1, min=0)
        lat_visible = block_last[None, :] < ws[:, None]  # (T, n_blocks)

        # window + landmark exact scores: (B, T, Hq, W + G)
        landmark_pos = torch.arange(n_blocks, device=device) * block_size
        lm_k = k[:, landmark_pos]  # (B, n_blocks, Hkv, Dv)
        lm_v = v[:, landmark_pos]
        lm_visible = lat_visible  # landmark visible iff its block is completed below window

        neg = torch.finfo(dtype).min
        # grouped matmul over kv heads: only (B, T, Hq, keys) logits materialize,
        # never the (B, T, Hq, W, D) product tensor
        # reshape not view: q arrives as a transpose(1, 2) of the HF layout and
        # splitting the head dim on a non-contiguous tensor needs the copy
        q_g = q.reshape(B, T, Hkv, group, D)
        # window scores via gather; early queries mask out pre-0 wrap positions.
        # Gather and value matmul run in 1024-row blocks: the full window
        # gather at 8k is an ~8.2GB live tensor per layer, enough to OOM a
        # 96GB card during backward even with the exact tiers detached.
        w_off = idx[:, None] - window + 1 + torch.arange(window, device=device)[None, :]
        w_valid = (w_off >= ws[:, None]) & (w_off <= idx[:, None])  # (T, W)
        w_off = w_off.clamp(min=0)
        row_chunk = 1024
        sw_parts = []
        for s in range(0, T, row_chunk):
            e = min(s + row_chunk, T)
            kw_c = k[:, w_off[s:e]].permute(0, 1, 3, 4, 2)  # (B, C, Hkv, Dv, W)
            sw_c = torch.matmul(q_g[:, s:e], kw_c) / s_d  # (B, C, Hkv, group, W)
            sw_parts.append(sw_c.view(B, e - s, Hq, window))
        sw = torch.cat(sw_parts, dim=1)
        sw = sw.masked_fill(~w_valid[None, :, None, :], neg)
        lm_kp = lm_k.permute(0, 2, 3, 1)  # (B, Hkv, Dv, n_blocks)
        sl = torch.matmul(q_g, lm_kp.unsqueeze(1)) / s_d  # (B, T, Hkv, group, n_blocks)
        sl = sl.view(B, T, Hq, n_blocks)
        sl = sl.masked_fill(~lm_visible[None, :, None, :], neg)

        logits_exact = torch.cat([sw, sl], dim=-1)  # (B, T, Hq, W + n_blocks)
    if L:
        q_flat = q.reshape(B, T, Hq * D)
        k_flat_mean = k_mean.reshape(B, n_blocks, Hkv * Dv)
        # Latent logits in at least fp32: untrained latents sit at ~660 scale
        # where bf16 ULP is ~4, so sub-4 relative differences (the ranking
        # signal) round away - the root cause of the flaky RULER seeds.
        # promote_types keeps the fp64 reference-test path in fp64.
        acc_dtype = torch.promote_types(q.dtype, torch.float32)
        q_lat = (q_flat @ w_q_lat).to(acc_dtype)  # (B, T, L)
        k_lat = (k_flat_mean @ w_k_lat).to(acc_dtype)  # (B, n_blocks, L)
        slt = (
            torch.matmul(
                q_lat.view(B, T, 1, 1, L), k_lat.permute(0, 2, 1).unsqueeze(1).unsqueeze(1)
            )
            / s_l
        )
        slt = slt.view(B, T, n_blocks)
        fmin = torch.finfo(acc_dtype).min
        slt = slt.masked_fill(~lat_visible, fmin)
        if latent_zscore:
            # Per-(query, row) standardization over visible blocks: keeps the
            # ranking, rescales the collapsed ~660 magnitude down to unit
            # scale so the latent tier competes with window/landmark logits
            # instead of drowning them. Noise latents (small spread) become
            # negligible; signal latents keep their ordering.
            vis = lat_visible.to(acc_dtype)
            cnt = vis.sum(-1, keepdim=True).clamp(min=1)
            mean = (slt * vis).sum(-1, keepdim=True) / cnt
            var = (((slt - mean) * vis) ** 2).sum(-1, keepdim=True) / cnt
            z = (slt - mean) / (var.sqrt() + 1e-6)
            slt = torch.where(lat_visible, z, torch.tensor(fmin, device=slt.device))
        if latent_clamp is not None:
            slt = slt.clamp(-latent_clamp, latent_clamp)
        slt_b = slt[:, :, None, :].expand(B, T, Hq, n_blocks)
        logits = torch.cat([logits_exact, slt_b], dim=-1).to(acc_dtype)
    else:
        logits = logits_exact.to(torch.promote_types(q.dtype, torch.float32))

    probs = torch.softmax(logits, dim=-1).to(dtype)
    pw = probs[..., :window].view(B, T, Hkv, group, window)
    plm = probs[..., window : window + n_blocks].view(B, T, Hkv, group, n_blocks)
    lm_vp = lm_v.permute(0, 2, 1, 3)  # (B, Hkv, n_blocks, Dv)
    out_parts = []
    for s in range(0, T, row_chunk):
        e = min(s + row_chunk, T)
        with torch.no_grad():
            vw_c = v[:, w_off[s:e]].permute(0, 1, 3, 2, 4)  # (B, C, Hkv, W, Dv)
        out_parts.append(torch.matmul(pw[:, s:e], vw_c))  # (B, C, Hkv, group, Dv)
    out = torch.cat(out_parts, dim=1)
    out = out + torch.matmul(plm, lm_vp.unsqueeze(1))
    out = out.view(B, T, Hq, Dv)
    if L:
        plt = probs[..., window + n_blocks :].view(B, T, Hkv, group, n_blocks)
        vm = v_mean.permute(0, 2, 1, 3)  # (B, Hkv, n_blocks, Dv)
        out = out + torch.matmul(plt, vm.unsqueeze(1)).view(B, T, Hq, Dv)
    logger.debug(
        "three-tier fast B=%d T=%d Hq=%d window=%d blocks=%d latent=%d",
        B,
        T,
        Hq,
        window,
        n_blocks,
        L,
    )
    aux = {"k_mean": k_mean, "v_mean": v_mean, "lat_visible": lat_visible}
    if return_probs_lat:
        aux["p_lat"] = probs[..., window + n_blocks :]
    return out, aux


class _TierTrainFn(torch.autograd.Function):
    # Training-path three-tier attention: forward runs the exact kernel
    # under no_grad and saves only the small tensors the analytic latent
    # gradient needs (q, out, p_lat, k_mean - ~1.4GB/layer at 32k instead
    # of the ~30GB saved+transient of the graph-mode softmax path). The
    # gradient wrt the latent logits is the exact softmax row gradient
    # restricted to the latent entries:
    #   dL_b = p_Lb * (<dOut, v_Lb> - <dOut, out>)
    # with the per-head contributions summed (one latent logit per
    # (batch, query, block), broadcast across heads). q/k/v grads return
    # None: the exact tiers are already computed under no_grad (cut
    # cross-layer path ruling), so no trainable parameter sits behind
    # them and the cut stream changes no trained weight.
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        w_q_lat,
        w_k_lat,
        window,
        block_size,
        q_offset,
        block_target=None,
        block_weight=1.0,
    ):
        with torch.no_grad():
            out, aux = three_tier_attention_fast(
                q,
                k,
                v,
                window,
                block_size,
                w_q_lat=w_q_lat,
                w_k_lat=w_k_lat,
                q_offset=q_offset,
                return_probs_lat=True,
            )
        ctx.save_for_backward(q, out, aux["p_lat"], aux["k_mean"], aux["v_mean"])
        ctx.w_q_lat = w_q_lat
        ctx.w_k_lat = w_k_lat
        ctx.dims = (window, block_size, q_offset)
        ctx.block_target = block_target
        ctx.block_weight = block_weight
        ctx.tabs = k.shape[1]
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, out, p_lat, k_mean, v_mean = ctx.saved_tensors
        w_q_lat, w_k_lat = ctx.w_q_lat, ctx.w_k_lat
        B, T, Hq, D = q.shape
        Hkv = k_mean.shape[2]
        Dv = v_mean.shape[-1]
        L = w_k_lat.shape[-1]
        group = Hq // Hkv
        n_blocks = k_mean.shape[1]

        dOut = grad_out.float()
        vm_e = v_mean.permute(0, 2, 1, 3)  # (B, Hkv, n_blocks, Dv)
        vm_e = vm_e.unsqueeze(2).expand(B, Hkv, group, n_blocks, Dv).reshape(B, Hq, n_blocks, Dv)
        term1 = torch.einsum("bthd,bhnd->bthn", dOut, vm_e.float())  # (B, T, Hq, n_blocks)
        term2 = (dOut * out.float()).sum(-1)  # (B, T, Hq)
        dL = p_lat.float() * (term1 - term2.unsqueeze(-1))
        dslt = dL.sum(dim=2)  # (B, T, n_blocks) - latent logit is head-broadcast

        q_flat = q.reshape(B, T, Hq * D)
        k_flat_mean = k_mean.reshape(B, n_blocks, Hkv * Dv)
        q_lat = (q_flat @ w_q_lat).float()  # (B, T, L)
        k_lat = (k_flat_mean @ w_k_lat).float()  # (B, n_blocks, L)

        s_l = L**0.5
        if ctx.block_target is not None:
            # Stage B auxiliary supervision: CE between the latent block
            # distribution and the teacher's per-block attention mass at the
            # sampled query rows only (block_target = (rows, target)). The
            # block softmax runs over the VISIBLE subset (masked exp
            # normalized, no fmin fill): a fmin-masked full softmax saturates
            # to one-hot at untrained latents and kills the gradient. The
            # latent logit is head-broadcast, so the per-head CE grad sums
            # to Hq * (p_block - target); unsampled and no-visible-block
            # rows (target 0) get zero aux grad.
            rows, tgt = ctx.block_target
            slt = torch.einsum("btl,bnl->btn", q_lat, k_lat) / s_l
            Tq = q.shape[1]
            idx = ctx.dims[2] + torch.arange(Tq, device=slt.device)
            ws = (idx - ctx.dims[0] + 1).clamp(min=0)
            nb = k_mean.shape[1]
            block_last = torch.tensor(
                [min((b + 1) * ctx.dims[1], ctx.tabs) - 1 for b in range(nb)], device=slt.device
            )
            vis = block_last[None, :] < ws[:, None]
            # max-shift before exp: trained latent logits reach ~660 where
            # raw exp overflows fp32 to inf (torch.softmax shifts internally;
            # this manual subset softmax must too)
            neg = torch.finfo(torch.float32).min
            slt_m = slt.masked_fill(~vis[None], neg)
            mx = slt_m.max(-1, keepdim=True).values
            exps = (slt_m - mx).exp() * vis[None]
            p_block = exps / exps.sum(-1, keepdim=True).clamp_min(1e-8)
            aux = torch.zeros_like(dslt)
            aux = aux.index_copy(
                1, rows, p_block[:, rows] - tgt.to(p_block.dtype)[None].expand(B, -1, -1)
            )
            dslt = dslt + ctx.block_weight * Hq * aux
        dq_lat = torch.einsum("btn,bnl->btl", dslt, k_lat) / s_l  # (B, T, L)
        dk_lat = torch.einsum("btn,btl->bnl", dslt, q_lat) / s_l  # (B, n_blocks, L)

        dw_q_lat = torch.zeros_like(w_q_lat, dtype=torch.float32)
        row_chunk = 4096
        for s in range(0, T, row_chunk):
            e = min(s + row_chunk, T)
            dw_q_lat += q_flat[s:e].float().reshape(-1, Hq * D).T @ dq_lat[s:e].reshape(-1, L)
        dw_k_lat = k_flat_mean.reshape(-1, Hkv * Dv).float().T @ dk_lat.reshape(-1, L).float()

        return (
            None,
            None,
            None,
            dw_q_lat.to(w_q_lat.dtype),
            dw_k_lat.to(w_k_lat.dtype),
            None,
            None,
            None,
            None,
            None,
        )


def three_tier_attention_train(
    q, k, v, window, block_size, w_q_lat, w_k_lat, q_offset=0, block_target=None, block_weight=1.0
):
    # Grad-enabled entry for the training forward; returns the same
    # (out, aux) shape as three_tier_attention_fast so tier_forward can
    # switch on grad mode without touching call sites. block_target (T,
    # n_blocks): teacher per-block attention mass for the Stage B aux CE.
    out = _TierTrainFn.apply(
        q, k, v, w_q_lat, w_k_lat, window, block_size, q_offset, block_target, block_weight
    )
    return out, {}
