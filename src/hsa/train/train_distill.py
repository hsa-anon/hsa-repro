import argparse
import contextlib
import json
import logging
import os
import time

import numpy as np
import torch

from hsa.train.qwen3_tier import (
    attach_tier_weights,
    disable_tier_mode,
    enable_tier_mode,
    full_attn_layers,
    trainable_parameters,
)

logger = logging.getLogger(__name__)


def load_bins(data_dir: str, seq_len: int):
    bins = []
    for f in sorted(os.listdir(data_dir)):
        if f.startswith("bin_") and f.endswith(".npy"):
            arr = np.load(os.path.join(data_dir, f))
            if arr.shape[1] >= seq_len + 1:
                bins.append(arr[:, : seq_len + 1])
                logger.info("loaded %s shape=%s", f, arr.shape)
    if not bins:
        raise RuntimeError(f"no usable bins in {data_dir} for seq_len={seq_len}")
    return np.concatenate(bins, axis=0)


def load_init_ckpt(model, ckpt_path: str, device: str) -> int:
    # Continue-training entry: load a previous tier-weights checkpoint
    # (same keying as the train loop's saves - GLOBAL decoder-layer index,
    # only layers with self_attn) into the freshly attached weights.
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    tw = ck["tier_weights"]
    # CausalLM wrappers nest the decoder (model.model.layers); bare decoders
    # expose .layers directly.
    layers = model.layers if hasattr(model, "layers") else model.model.layers
    loaded = 0
    for n, layer in enumerate(layers):
        if not hasattr(layer, "self_attn"):
            continue
        wq = tw.get(f"layer{n}.w_q_lat")
        wk = tw.get(f"layer{n}.w_k_lat")
        if wq is None or wk is None:
            raise RuntimeError(f"init ckpt missing tier weights for layer {n}")
        layer.self_attn.w_q_lat.data = wq.to(device=device, dtype=layer.self_attn.w_q_lat.dtype)
        layer.self_attn.w_k_lat.data = wk.to(device=device, dtype=layer.self_attn.w_k_lat.dtype)
        loaded += 1
    logger.info("init ckpt loaded into %d layers from %s", loaded, ckpt_path)
    return loaded


def activation_offload_hooks(threshold_bytes=200 * 1024 * 1024):
    # Offload large saved activations (the 64 per-layer checkpoint inputs,
    # ~26GB at seq 32k) to host RAM during forward and pull them back during
    # backward. Model-agnostic: saved_tensors_hooks applies to whatever the
    # autograd graph saves. Only non-leaf tensors above the threshold are
    # packed - leaves are frozen weights (55GB, re-fetched constantly during
    # checkpoint recompute, far too hot to move). Math is unchanged; this only
    # relocates storage.
    stats = {"packed": 0, "bytes": 0, "unpacked": 0}

    def pack(t):
        if not t.is_leaf and t.untyped_storage().nbytes() > threshold_bytes:
            stats["packed"] += 1
            stats["bytes"] += t.untyped_storage().nbytes()
            return t.to("cpu")
        return t

    def unpack(t):
        if t.device.type == "cpu":
            stats["unpacked"] += 1
            return t.to(DEVICE[0])
        return t

    class _LoggingHooks:
        def __enter__(self):
            self._cm = torch.autograd.graph.saved_tensors_hooks(pack, unpack)
            self._cm.__enter__()
            return self

        def __exit__(self, *exc):
            self._cm.__exit__(*exc)
            if MEM_PROBE[0]:
                logger.info(
                    "OFFLOAD packed=%d bytes=%.2fG unpacked=%d",
                    stats["packed"],
                    stats["bytes"] / 2**30,
                    stats["unpacked"],
                )
            return False

    return _LoggingHooks()


DEVICE = ["cpu"]
MEM_PROBE = [False]


def _mem_log(tag):
    # Phase-level NPU memory probe (enabled by --mem-probe): logs allocated,
    # reserved and peak after each named phase so a long-seq OOM can be
    # attributed to the phase that actually owns the memory.
    if not MEM_PROBE[0]:
        return
    if hasattr(torch, "npu") and torch.npu.is_available():
        alloc = torch.npu.memory_allocated() / 2**30
        reserved = torch.npu.memory_reserved() / 2**30
        peak = torch.npu.max_memory_allocated() / 2**30
    elif torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 2**30
        reserved = torch.cuda.memory_reserved() / 2**30
        peak = torch.cuda.max_memory_allocated() / 2**30
    else:
        return
    logger.info("MEM %s alloc=%.2fG reserved=%.2fG peak=%.2fG", tag, alloc, reserved, peak)


def _leak_top(tag, topn=10):
    # GC-based live-tensor census (enabled by --mem-probe): scans all Python-
    # reachable tensors on the training device and aggregates bytes by
    # (shape, dtype). Catches retention that saved_tensors_hooks cannot see -
    # python-held references, module attributes, closure captures. Expensive:
    # only call at sparse probe points.
    if not MEM_PROBE[0]:
        return
    import gc
    from collections import Counter

    gc.collect()
    dev = DEVICE[0].split(":")[0]
    census = Counter()
    n_tensors = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.device.type == dev:
                n_tensors += 1
                census[(tuple(obj.shape), str(obj.dtype))] += obj.numel() * obj.element_size()
        except Exception as exc:  # noqa: BLE001 - gc proxy objects can raise arbitrary errors
            logger.debug("LEAK %s gc scan skipped an object: %s", tag, exc)
            continue
    logger.info("LEAK %s live_tensors=%d total=%.2fG", tag, n_tensors, sum(census.values()) / 2**30)
    for key, byts in census.most_common(topn):
        logger.info("LEAK %s %s %s %.3fG", tag, key[0], key[1], byts / 2**30)


def peeled_forward_store(model, ids):
    # Peeled mode forward: run the 64 decoder layers one by one under
    # no_grad, stashing every layer input on host RAM (~336MB each at 32k,
    # 21.5G total). Replaces checkpointing entirely: the backward pass
    # (peeled_backward) recomputes ONE layer at a time under grad mode with
    # an explicitly scoped graph, so no autograd-engine accumulation across
    # layers is possible - the structural fix for the 0.475G/layer backward
    # climb that killed every checkpointed 32k attempt.
    import transformers.models.qwen3_5.modeling_qwen3_5 as mq

    m = model.model
    inputs_embeds = m.embed_tokens(ids)
    position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
    position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
    text_position_ids = position_ids[0]
    position_ids = position_ids[1:]
    mask_kwargs = {
        "config": m.config,
        "inputs_embeds": inputs_embeds,
        "attention_mask": None,
        "past_key_values": None,
        "position_ids": text_position_ids,
    }
    cache = {
        "mask_mapping": {
            "full_attention": mq.create_causal_mask(**mask_kwargs),
            "linear_attention": mq.create_recurrent_attention_mask(**mask_kwargs),
        },
        "position_embeddings": m.rotary_emb(inputs_embeds, position_ids),
        "text_position_ids": text_position_ids,
        "layers": list(m.layers)[: m.config.num_hidden_layers],
        "layer_types": m.config.layer_types,
        "m": m,
    }
    stored = [inputs_embeds.to("cpu")]
    h = inputs_embeds
    with torch.no_grad():
        for i, layer in enumerate(cache["layers"]):
            mi = cache["mask_mapping"][cache["layer_types"][i]]
            h = layer(
                h,
                position_embeddings=cache["position_embeddings"],
                attention_mask=mi,
                position_ids=text_position_ids,
                past_key_values=None,
                use_cache=False,
            )
            stored.append(h.to("cpu"))
    _mem_log("peeled-forward-end")
    return stored, cache


def peeled_backward(model, stored, cache, g_last):
    # Peeled mode backward: gradient g_last (wrt the final norm output,
    # full sequence length) is backpropagated through the norm, then
    # layer-by-layer from the top. Each iteration rebuilds exactly one
    # layer's graph under grad mode, backprops into the stored layer input
    # (a fresh leaf), and releases the graph - the next iteration only
    # carries the (1, T, 5120) gradient. Math is identical to a single
    # whole-network backward: same recompute, same chain rule, tier
    # weights accumulate .grad the same way.
    m = cache["m"]
    dev = next(model.parameters()).device
    h64 = stored[-1].to(dev).detach().requires_grad_(True)
    s_hidden = m.norm(h64)
    s_hidden.backward(gradient=g_last)
    g = h64.grad
    h64.grad = None
    del s_hidden, h64
    layers = cache["layers"]
    for i in reversed(range(len(layers))):
        h_i = stored[i].to(dev).detach().requires_grad_(True)
        mi = cache["mask_mapping"][cache["layer_types"][i]]
        h_out = layers[i](
            h_i,
            position_embeddings=cache["position_embeddings"],
            attention_mask=mi,
            position_ids=cache["text_position_ids"],
            past_key_values=None,
            use_cache=False,
        )
        h_out.backward(gradient=g)
        g = h_i.grad
        h_i.grad = None
        del h_out, h_i
        if MEM_PROBE[0] and i % 8 == 0:
            _mem_log(f"peel-layer={i}")
    _mem_log("peel-end")


def pair_checkpointed_forward(model, ids):
    # Manual reentrant per-layer checkpointing for the 32k stage. HF's
    # non-reentrant per-layer checkpointing holds every layer input
    # on-device AND its internal saved_tensors_hooks shadow any outer
    # offload hook, so this path replicates the Qwen3_5TextModel.forward
    # setup and checkpoints EACH decoder layer with use_reentrant=True:
    # reentrant checkpoint does not shadow the outer hooks, so both the
    # layer inputs and the big recomputed tier-attention saved tensors
    # pack to host RAM (measured: post-forward 61.6G -> 51.3G). Finer
    # segments also halve the live grad-mode recompute transient, which
    # was the actual 32k OOM driver at pair granularity (~38G per pair).
    import transformers.models.qwen3_5.modeling_qwen3_5 as mq
    from torch.utils.checkpoint import checkpoint

    m = model.model
    inputs_embeds = m.embed_tokens(ids)
    position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
    position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
    text_position_ids = position_ids[0]
    position_ids = position_ids[1:]
    mask_kwargs = {
        "config": m.config,
        "inputs_embeds": inputs_embeds,
        "attention_mask": None,
        "past_key_values": None,
        "position_ids": text_position_ids,
    }
    causal_mask_mapping = {
        "full_attention": mq.create_causal_mask(**mask_kwargs),
        "linear_attention": mq.create_recurrent_attention_mask(**mask_kwargs),
    }
    position_embeddings = m.rotary_emb(inputs_embeds, position_ids)
    layer_types = m.config.layer_types
    layers = list(m.layers)[: m.config.num_hidden_layers]
    h = inputs_embeds.detach().requires_grad_(True)
    for i, layer in enumerate(layers):
        mi = causal_mask_mapping[layer_types[i]]

        def seg(h, layer=layer, mi=mi):
            return layer(
                h,
                position_embeddings=position_embeddings,
                attention_mask=mi,
                position_ids=text_position_ids,
                past_key_values=None,
                use_cache=False,
            )

        h = checkpoint(seg, h, use_reentrant=True)
        if MEM_PROBE[0] and i % 8 == 0:

            def _probe_hook(g, i=i):
                _mem_log(f"bwd-layer={i}")
                _leak_top(f"bwd-layer={i}")

            h.register_hook(_probe_hook)
        if MEM_PROBE[0] and (i + 1) % 16 == 0:
            _mem_log(f"layer-seg={i + 1}/64")
    _mem_log("pair-forward-end")
    return m.norm(h)


def chunked_ce_grad(lm_head, hidden_leaf, labels, chunk, ce_sum_fn, weights=None):
    # Chunked lm_head cross-entropy with per-chunk backward: full-sequence
    # fp32 logits are ~10GB per 16k tokens at vocab 152k, which OOMs the 96GB
    # card at 16k and is hopeless at 32k. Splitting into chunks keeps only one
    # chunk's logits live; gradients accumulate into hidden_leaf.grad and the
    # result is bit-equivalent math to the full-sequence CE mean.
    # weights (optional): per-label-position weights (1, T) aligned with
    # labels; loss becomes the weighted mean over tokens (needle-boost).
    total = torch.zeros((), dtype=torch.float32, device=hidden_leaf.device)
    T = hidden_leaf.shape[1]
    # global weight sum keeps chunked weighted CE equal in scale to the
    # unweighted path: each chunk contributes sum(w*ce)/W, and the per-chunk
    # sums add up to the single global weighted mean
    W = weights.float().sum().clamp_min(1.0) if weights is not None else None
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        logits_c = lm_head(hidden_leaf[:, s:e])
        if weights is not None:
            ce_pt = torch.nn.functional.cross_entropy(
                logits_c.reshape(-1, logits_c.shape[-1]).float(),
                labels[:, s:e].reshape(-1),
                reduction="none",
            )
            w = weights[:, s:e].reshape(-1).float()
            ce_c = (ce_pt * w).sum() / W
        else:
            ce_c = (
                ce_sum_fn(
                    logits_c.reshape(-1, logits_c.shape[-1]).float(), labels[:, s:e].reshape(-1)
                )
                / T
            )
        ce_c.backward()
        total += ce_c.detach()
    return float(total), hidden_leaf.grad


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--d-latent", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--ckpt-every", type=int, default=200)
    parser.add_argument("--device", default="")
    parser.add_argument(
        "--init-ckpt", default="", help="continue-training tier weights from this checkpoint"
    )
    parser.add_argument(
        "--no-teacher",
        action="store_true",
        help="skip the dense teacher pass; student CE loss only",
    )
    parser.add_argument(
        "--loss-chunk", type=int, default=4096, help="lm_head CE chunk size in tokens"
    )
    parser.add_argument(
        "--offload-activations",
        action="store_true",
        help="offload large saved activations to host RAM (long-seq stages)",
    )
    parser.add_argument(
        "--offload-threshold",
        type=int,
        default=200 * 1024 * 1024,
        help="saved-tensor offload threshold in bytes",
    )
    parser.add_argument(
        "--ckpt-pairs",
        action="store_true",
        help="checkpoint decoder-layer pairs instead of single layers (32k stage)",
    )
    parser.add_argument(
        "--peeled",
        action="store_true",
        help="peeled forward/backward: store layer inputs on host, recompute+backward one layer at a time (32k stage)",
    )
    parser.add_argument(
        "--stage-b",
        action="store_true",
        help="Phase 2b Stage B: auxiliary CE on the latent block softmax against teacher per-block attention mass",
    )
    parser.add_argument("--block-weight", type=float, default=1.0, help="Stage B aux loss weight")
    parser.add_argument(
        "--block-rows",
        type=int,
        default=64,
        help="Stage B: teacher query rows sampled per step for block-mass targets",
    )
    parser.add_argument(
        "--mem-probe", action="store_true", help="log per-phase NPU memory stats (OOM diagnosis)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.makedirs(args.out_dir, exist_ok=True)

    if args.stage_b and args.no_teacher:
        raise RuntimeError("--stage-b requires the dense teacher pass (drop --no-teacher)")
    if not args.device:
        args.device = (
            "npu"
            if torch.npu.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    DEVICE[0] = args.device
    MEM_PROBE[0] = args.mem_probe
    logger.info(
        "device=%s steps=%d seq=%d window=%d block=%d",
        args.device,
        args.steps,
        args.seq_len,
        args.window,
        args.block_size,
    )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model.to(args.device)
    if args.ckpt_pairs or args.peeled:
        # ckpt-pairs / peeled manage their own recomputation regions
        model.gradient_checkpointing_disable()
    else:
        model.gradient_checkpointing_enable()
    # gradient checkpointing only engages when the module tree is in
    # training mode; without this every layer keeps its backward graph
    # resident and 27B OOMs immediately
    model.train()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    attach_tier_weights(model, args.d_latent)
    for p in model.parameters():
        if p.device.type != args.device.split(":")[0]:
            p.data = p.data.to(args.device)
    if args.init_ckpt:
        load_init_ckpt(model, args.init_ckpt, args.device)
    trainable = trainable_parameters(model)
    logger.info("trainable params: %.1fM", sum(p.numel() for p in trainable) / 1e6)
    if args.stage_b:
        from hsa.train.block_mass import (
            capture_teacher_attn_inputs,
            compute_block_mass_targets,
            release_capture_hooks,
        )

        tier_attns = [l.self_attn for l in full_attn_layers(model)]
        for a in tier_attns:
            a._block_weight = args.block_weight
        logger.info(
            "stage-b enabled: block-weight=%.3f rows=%d", args.block_weight, args.block_rows
        )

    data = load_bins(args.data_dir, args.seq_len)
    input_ids_all = torch.tensor(
        np.ascontiguousarray(data[:, : args.seq_len]), dtype=torch.long, device=args.device
    )
    labels_all = torch.tensor(
        np.ascontiguousarray(data[:, 1 : args.seq_len + 1]), dtype=torch.long, device=args.device
    )
    # optional per-position CE weights (needle-boosted training, Phase 2b
    # stage A): data_dir/wgt_{seq_len}.npy, shape (N, seq_len), aligned with
    # labels. Absent file = uniform weights, unchanged behavior.
    wgt_path = os.path.join(args.data_dir, f"wgt_{args.seq_len}.npy")
    if os.path.exists(wgt_path):
        wgt = np.load(wgt_path)
        if wgt.shape[0] != data.shape[0]:
            raise RuntimeError(
                f"weight rows {wgt.shape[0]} != data rows {data.shape[0]} in {wgt_path}"
            )
        wgt_all = torch.tensor(
            np.ascontiguousarray(wgt[:, : args.seq_len]), dtype=torch.float32, device=args.device
        )
        logger.info(
            "loaded label weights %s (boost fraction %.4f)",
            wgt_path,
            float((wgt[:, : args.seq_len] > 1.0).mean()),
        )
    else:
        wgt_all = None
        logger.info("no label weights at %s - uniform CE", wgt_path)

    teacher_ce_sum = torch.nn.CrossEntropyLoss(reduction="sum")
    student_ce_sum = torch.nn.CrossEntropyLoss(reduction="sum")
    distill_fn = torch.nn.MSELoss()
    opt = torch.optim.Adam(trainable, lr=args.lr)

    history = []
    rng = np.random.default_rng(7)
    _mem_log("after-data-setup")
    t0 = time.time()
    for step in range(args.steps):
        i = int(rng.integers(0, input_ids_all.shape[0]))
        ids = input_ids_all[i : i + 1]

        # teacher on the SAME weights: tier mode off, frozen, no grad.
        # --no-teacher skips this entirely (long-seq stages where the dense
        # teacher pass OOMs or dominates step time; matrix variant axis).
        # Backbone-only forward: no full-sequence lm_head logits, no 64-layer
        # hidden_states stack - both are fatal at 16k/32k. CE runs chunked.
        if args.no_teacher:
            t_loss = float("nan")
            t_hidden = None
        else:
            disable_tier_mode(model)
            cap_handles = None
            if args.stage_b:
                captures, cap_handles = capture_teacher_attn_inputs(model)
            with torch.no_grad():
                t_hidden = model.model(ids).last_hidden_state
                n_pos = args.seq_len - 1
                t_sum = torch.zeros((), dtype=torch.float32, device=args.device)
                for s in range(0, n_pos, args.loss_chunk):
                    e = min(s + args.loss_chunk, n_pos)
                    t_logits_c = model.lm_head(t_hidden[:, s:e])
                    t_sum = (
                        t_sum
                        + teacher_ce_sum(
                            t_logits_c.reshape(-1, t_logits_c.shape[-1]).float(),
                            labels_all[i : i + 1, s:e].reshape(-1),
                        ).detach()
                    )
                t_loss = float(t_sum) / n_pos
                if args.stage_b:
                    # rows with >= 2 visible blocks only: a single-visible-
                    # block row's subset softmax is one-hot and carries no
                    # aux gradient signal
                    rows = rng.integers(
                        args.window + 2 * args.block_size, args.seq_len, size=args.block_rows
                    )
                    bmass = compute_block_mass_targets(
                        model,
                        captures,
                        rows,
                        args.window,
                        args.block_size,
                        torch.device(args.device),
                    )
                    for attn, (rows_t, tgt, _ent) in bmass.items():
                        attn._block_target = (rows_t, tgt)
            if cap_handles is not None:
                release_capture_hooks(cap_handles)
            if hasattr(torch, "npu") and torch.npu.is_available():
                torch.npu.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

        # student: tier mode on, injected params train. CE runs chunk-by-chunk
        # on a detached leaf; each chunk backward frees its logits graph, then
        # one backbone backward pushes the accumulated hidden gradient through
        # the checkpointed layers (single recompute, same gradient math).
        enable_tier_mode(model, args.window, args.block_size)
        ctx = (
            activation_offload_hooks(args.offload_threshold)
            if args.offload_activations
            else contextlib.nullcontext()
        )
        with ctx:
            if MEM_PROBE[0] and hasattr(torch, "npu") and torch.npu.is_available():
                torch.npu.reset_peak_memory_stats()
            _mem_log(f"step={step} pre-forward")
            if args.peeled:
                stored, pcache = peeled_forward_store(model, ids)
                h64 = stored[-1].to(model.model.norm.weight.device).detach().requires_grad_(True)
                s_hidden = model.model.norm(h64)
            elif args.ckpt_pairs:
                s_hidden = pair_checkpointed_forward(model, ids)
            else:
                s_hidden = model.model(ids).last_hidden_state
            _mem_log(f"step={step} post-forward")
            h_leaf = s_hidden[:, :-1].detach().requires_grad_(True)
            labels_i = labels_all[i : i + 1, :-1]
            s_loss, h_grad = chunked_ce_grad(
                model.lm_head,
                h_leaf,
                labels_i,
                args.loss_chunk,
                student_ce_sum,
                weights=wgt_all[i : i + 1, :-1] if wgt_all is not None else None,
            )
            if t_hidden is not None:
                mse_val = distill_fn(h_leaf, t_hidden[:, :-1])
                mse_grad = torch.autograd.grad(mse_val, h_leaf)[0]
                loss = float(mse_val.detach()) + s_loss
                g = h_grad + mse_grad
            else:
                loss = s_loss
                g = h_grad
            _mem_log(f"step={step} post-ce")
            opt.zero_grad()
            if args.peeled:
                g_full = torch.zeros(
                    1, s_hidden.shape[1], s_hidden.shape[2], dtype=g.dtype, device=g.device
                )
                g_full[:, :-1] = g
                del s_hidden, h_leaf, h64
                peeled_backward(model, stored, pcache, g_full)
                del stored, pcache, g_full
            else:
                s_hidden[:, :-1].backward(gradient=g)
            _mem_log(f"step={step} post-backward")
        opt.step()
        if args.stage_b:
            for a in tier_attns:
                a._block_target = None
        _mem_log(f"step={step} post-step")
        history.append(
            {
                "step": step,
                "loss": float(loss),
                "teacher_lm": float(t_loss),
                "student_lm": float(s_loss),
            }
        )
        if step % args.log_every == 0:
            ent_sfx = ""
            if args.stage_b:
                ent_sfx = " b_ent=%.4f" % (sum(e for _, _, e in bmass.values()) / len(bmass))
            logger.info(
                "step=%d loss=%.6f t_lm=%.4f s_lm=%.4f%s elapsed=%.0fs",
                step,
                float(loss),
                float(t_loss),
                float(s_loss),
                ent_sfx,
                time.time() - t0,
            )
        if (step + 1) % args.ckpt_every == 0 or step == args.steps - 1:
            ck = {
                f"layer{n}.{k}": v.detach().cpu()
                for n, layer in enumerate(model.model.layers)
                if hasattr(layer, "self_attn")
                for k, v in (
                    ("w_q_lat", layer.self_attn.w_q_lat),
                    ("w_k_lat", layer.self_attn.w_k_lat),
                )
            }
            torch.save(
                {"tier_weights": ck, "config": vars(args), "history": history},
                os.path.join(args.out_dir, f"ckpt_{step + 1}.pt"),
            )
            logger.info("checkpoint saved step=%d", step + 1)

    with open(os.path.join(args.out_dir, "train_log.json"), "w") as f:
        json.dump({"config": vars(args), "history": history}, f, indent=2)
    logger.info("done steps=%d final_loss=%.6f", args.steps, history[-1]["loss"])


if __name__ == "__main__":
    main()
