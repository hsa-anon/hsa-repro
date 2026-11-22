import argparse
import json
import logging
import random
import time

import torch

from hsa.train.qwen3_tier import attach_tier_weights, enable_tier_mode, full_attn_layers

logger = logging.getLogger(__name__)

FILLER = "The sun rises in the east and sets in the west. Grass is green in the summer. "
NEEDLE_TMPL = "One of the special magic numbers for {key} is: {value}."

# --- RULER VT (variable tracking): chained assignments, ask for the tail ---
VT_CHAIN_TEMPLATE = "VAR_{tag}{i} = {value}. "
VT_HOP_TEMPLATE = "VAR_{tag}{i} = VAR_{tag}{prev}. "
VT_QUESTION_TEMPLATE = "What is the value of VAR_{tag}{n} mentioned in the provided text?"

# --- RULER FWE (frequent words extraction): coded words outnumber fillers ---
FWE_QUESTION_TEMPLATE = (
    "What are the 3 most frequent words in the provided text? "
    "Answer with the 3 words separated by commas."
)

# --- RULER QA: one fact-bearing passage among distractor essays ---
QA_FACT_TEMPLATE = (
    "The {key} partnership agreement was signed in {city}, and the "
    "ceremony lasted exactly {days} days. "
)
QA_QUESTION_TEMPLATE = "In which city was the {key} partnership agreement signed?"
QA_CITIES = [
    "Ravensburg", "Thornbury", "Calderdale", "Mistelbach", "Ostrowiec",
    "Velenje", "Koprivnica", "Almelo", "Sonderborg", "Larvik",
]

# --- RULER CWE: 10 coded words with equal frequency, all must be named ---
CWE_QUESTION_TEMPLATE = (
    "What are the 10 most common words in the provided text? "
    "Answer with the 10 words separated by commas."
)


def _filler_words(rng, count):
    # Unique coded filler tokens: each appears exactly once, so repeated
    # coded words dominate by construction (matches the serving-path
    # generators in the original benchmark scripts).
    words = []
    seen = set()
    while len(words) < count:
        token = f"zelvar{rng.randint(100000, 999999)}"
        if token in seen:
            continue
        seen.add(token)
        words.append(token)
    return words


def build_task_prompt(task, target_tokens, seed):
    # Text-level generators mirroring the serving-path sweep; returns
    # (prompt_text, expected) where expected is a substring (vt/qa) or a
    # list of words all of which must appear (fwe/cwe).
    rng = random.Random(seed)
    if task == "vt":
        tag = f"grasveld{rng.randint(1000, 9999)}"
        value = f"{rng.randint(100000, 999999)}-{rng.randint(100, 999)}"
        hops = 4
        words = _filler_words(rng, target_tokens // 2)
        total = len(words)
        for i in range(1, hops + 1):
            piece = (
                VT_CHAIN_TEMPLATE.format(tag=tag, i=i, value=value)
                if i == 1
                else VT_HOP_TEMPLATE.format(tag=tag, i=i, prev=i - 1)
            )
            pos = int(total * i / (hops + 1))
            words[pos:pos] = piece.split()
        prompt = " ".join(words) + "\n\n" + VT_QUESTION_TEMPLATE.format(tag=tag, n=hops)
        return prompt, value
    if task == "fwe":
        coded = []
        while len(coded) < 3:
            word = f"krovex{rng.randint(100, 999)}"
            if word not in coded:
                coded.append(word)
        counts = [
            max(8, target_tokens // 40),
            max(6, target_tokens // 55),
            max(5, target_tokens // 80),
        ]
        filler_count = target_tokens // 2 - sum(counts)
        words = _filler_words(rng, max(1, filler_count))
        for word, count in zip(coded, counts):
            for _ in range(count):
                words.insert(rng.randrange(len(words) + 1), word)
        prompt = " ".join(words) + "\n\n" + FWE_QUESTION_TEMPLATE
        return prompt, coded
    if task == "qa":
        key = f"grasveld-{rng.randint(1000, 9999)}"
        gold_city = rng.choice(QA_CITIES)
        days = rng.randint(2, 9)
        other_cities = [c for c in QA_CITIES if c != gold_city]
        rng.shuffle(other_cities)
        gold_passage = QA_FACT_TEMPLATE.format(key=key, city=gold_city, days=days)
        distractors = 7
        passages = [
            QA_FACT_TEMPLATE.format(
                key=f"kelvor-{rng.randint(1000, 9999)}",
                city=other_cities[i % len(other_cities)],
                days=rng.randint(2, 9),
            )
            for i in range(distractors)
        ]
        passages.insert(rng.randrange(len(passages) + 1), gold_passage)
        filler_words = _filler_words(rng, max(1, target_tokens // 2 - 40 * len(passages)))
        third = max(1, len(filler_words) // (len(passages) + 1))
        chunks = []
        fill_pos = 0
        for passage in passages:
            chunks.append(" ".join(filler_words[fill_pos:fill_pos + third]))
            fill_pos += third
            chunks.append(passage)
        prompt = " ".join(chunks) + "\n\n" + QA_QUESTION_TEMPLATE.format(key=key)
        return prompt, gold_city
    if task == "cwe":
        common = 10
        coded = []
        while len(coded) < common:
            word = f"myrvik{rng.randint(100, 999)}"
            if word not in coded:
                coded.append(word)
        repeats = max(3, target_tokens // 200)
        filler_count = target_tokens // 2 - common * repeats
        words = _filler_words(rng, max(1, filler_count))
        for word in coded:
            for _ in range(repeats):
                words.insert(rng.randrange(len(words) + 1), word)
        prompt = " ".join(words) + "\n\n" + CWE_QUESTION_TEMPLATE
        return prompt, coded
    raise ValueError(f"unknown task {task}")


def build_calibrated(task, target_tokens, seed, tok):
    # Synthetic coded fillers tokenize to ~8 tokens per word (vs ~1.3 for
    # natural text), so the word-level budget overshoots ~4x. Rebuild with
    # a rescaled target until the measured token count is within 5% of the
    # requested budget; expected values are drawn before filler sizing, so
    # they are stable across rescales.
    target = target_tokens
    text, expected = build_task_prompt(task, target, seed)
    ids = tok(text, add_special_tokens=False).input_ids
    for _ in range(2):
        if abs(len(ids) - target_tokens) <= 0.05 * target_tokens:
            break
        target = max(1024, int(target * target_tokens / len(ids)))
        text, expected = build_task_prompt(task, target, seed)
        ids = tok(text, add_special_tokens=False).input_ids
    return text, expected


def build_prompt(tokenizer, length_tokens, key, value, depth_ratio, seed):
    # NIAH: filler haystack, one needle at depth_ratio, question at the end.
    random.Random(seed)
    needle = NEEDLE_TMPL.format(key=key, value=value)
    n_fill = length_tokens - len(tokenizer(needle).input_ids) - 32
    fill_ids = tokenizer(FILLER, add_special_tokens=False).input_ids
    haystack = []
    while sum(len(x) for x in haystack) < n_fill:
        haystack.append(fill_ids)
    flat = [i for chunk in haystack for i in chunk][:n_fill]
    cut = int(n_fill * depth_ratio)
    needle_ids = tokenizer(needle, add_special_tokens=False).input_ids
    question = "What is the special magic number for " + key + " mentioned in the provided text?"
    q_ids = tokenizer(question, add_special_tokens=False).input_ids
    return flat[:cut] + needle_ids + flat[cut:] + q_ids


def chunked_generate(model, input_ids, max_new_tokens, chunk, eos_token_id):
    # Chunked prefill + greedy decode for prompts too long for a single
    # dense forward. Keeps past_key_values across chunks (GDN layers own
    # the official HF cache); position_ids are passed explicitly because a
    # manual forward cannot infer them from the cache alone.
    device = input_ids.device
    T = input_ids.shape[1]
    past = None
    pos = 0
    with torch.no_grad():
        while pos < T:
            end = min(pos + chunk, T)
            pos_ids = torch.arange(pos, end, device=device).unsqueeze(0)
            out = model(
                input_ids[:, pos:end], past_key_values=past, use_cache=True, position_ids=pos_ids
            )
            past = out.past_key_values
            pos = end
        # the last prefill forward already consumed the final prompt token;
        # its last-position logits yield the first generated token
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        generated = [nxt]
        cur = nxt
        next_pos = T
        for _ in range(max_new_tokens - 1):
            pos_ids = torch.tensor([[next_pos]], device=device)
            out = model(cur, past_key_values=past, use_cache=True, position_ids=pos_ids)
            past = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            generated.append(nxt)
            cur = nxt
            next_pos += 1
            if eos_token_id is not None and int(nxt.item()) == eos_token_id:
                break
    return torch.cat([input_ids] + generated, dim=1)


def run_ruler(
    model_path,
    ckpt_path,
    out_path,
    lengths,
    depths,
    seeds,
    max_new_tokens,
    window=512,
    block_size=2048,
    dense=False,
    no_ckpt=False,
    latent_clamp=None,
    decode_topk=None,
    prefill_chunk=None,
    latent_zscore=False,
    decode_quest=None,
    task="niah",
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = (
        "npu"
        if getattr(torch, "npu", None) and torch.npu.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model.to(device)
    model.config.use_cache = True
    model.eval()

    attach_tier_weights(model, 2048)
    ck = None if no_ckpt else torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # ckpt keys carry the GLOBAL decoder-layer index (train loop enumerates
    # all layers, skips GatedDeltaNet); map by the same enumeration.
    loaded = 0
    if ck is not None:
        tw = ck["tier_weights"]
        for n, layer in enumerate(model.model.layers):
            if not hasattr(layer, "self_attn"):
                continue
            wq = tw.get(f"layer{n}.w_q_lat")
            wk = tw.get(f"layer{n}.w_k_lat")
            if wq is None or wk is None:
                raise RuntimeError(f"ckpt missing tier weights for layer {n}")
            layer.self_attn.w_q_lat.data = wq.to(device=device, dtype=layer.self_attn.w_q_lat.dtype)
            layer.self_attn.w_k_lat.data = wk.to(device=device, dtype=layer.self_attn.w_k_lat.dtype)
            loaded += 1
    logger.info("loaded tier weights into %d layers from %s", loaded, ckpt_path)
    if dense:
        logger.info("dense control run: tier mode NOT enabled")
    else:
        enable_tier_mode(model, window=window, block_size=block_size)
        for layer in full_attn_layers(model):
            layer.self_attn._tier_decode = True
            layer.self_attn._latent_clamp = latent_clamp
            layer.self_attn._latent_zscore = latent_zscore
            layer.self_attn._decode_topk = None if decode_quest else decode_topk
            layer.self_attn._decode_quest = decode_quest
        if decode_quest:
            logger.info(
                "quest external baseline: dense prefill + page-bound decode, pages=%d page_size=64",
                decode_quest,
            )

    keys = ["Fucheng", "Zhaoyun", "Wolong", "Erlang", "Changshan"]
    results = []
    import gc

    for length in lengths:
        for seed in seeds:
            # Per-case memory hygiene: drop the previous case's tier caches
            # (full-history k/v per layer) and trim the NPU allocator pool -
            # without this, reserved memory creeps ~2 GB per case until the
            # next prefill allocation OOMs.
            for layer in full_attn_layers(model):
                layer.self_attn._tier_cache = None
            gc.collect()
            if hasattr(torch, "npu"):
                torch.npu.empty_cache()
            depth = depths[seed % len(depths)]
            key = keys[seed % len(keys)]
            value = f"{rng_val(seed)}-{rng_val(seed + 17)}"
            if task == "niah":
                ids = build_prompt(tok, length, key, value, depth, seed)
                expected = value
            else:
                text, expected = build_calibrated(task, length, seed, tok)
                ids = tok(text, add_special_tokens=False).input_ids
                logger.info("task=%s built %d tokens (target %d)", task, len(ids), length)
            input_ids = torch.tensor([ids], device=device)
            t0 = time.time()
            with torch.no_grad():
                if prefill_chunk:
                    out = chunked_generate(
                        model, input_ids, max_new_tokens, prefill_chunk, tok.eos_token_id
                    )
                else:
                    out = model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tok.eos_token_id,
                    )
            answer = tok.decode(out[0, input_ids.shape[1] :], skip_special_tokens=True)
            hit = all(w in answer for w in expected) if isinstance(expected, list) else expected in answer
            elapsed = time.time() - t0
            results.append(
                {
                    "length": length,
                    "seed": seed,
                    "depth": depth,
                    "task": task,
                    "key": key,
                    "value": value,
                    "expected": expected,
                    "hit": hit,
                    "answer": answer.strip()[:200],
                    "prefill_plus_decode_s": round(elapsed, 1),
                }
            )
            logger.info(
                "len=%d seed=%d depth=%.2f hit=%s ans=%r %.0fs",
                length,
                seed,
                depth,
                hit,
                answer.strip()[:60],
                elapsed,
            )
    summary = {}
    for length in lengths:
        rows = [r for r in results if r["length"] == length]
        summary[str(length)] = {
            "hits": sum(r["hit"] for r in rows),
            "total": len(rows),
            "ratio": round(sum(r["hit"] for r in rows) / len(rows), 4),
        }
    payload = {
        "summary": summary,
        "details": results,
        "ckpt": ckpt_path,
        "window": window,
        "block_size": block_size,
        "dense": dense,
        "decode_quest": decode_quest,
        "decode_topk": decode_topk,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("summary: %s", json.dumps(summary))


def rng_val(seed):
    rng = random.Random(seed)
    return str(rng.randint(100000, 999999))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--ckpt", default=None, help="required unless --no-ckpt")
    p.add_argument("--out", required=True)
    p.add_argument("--lengths", default="4096,32768,131072")
    p.add_argument("--depths", default="0.25,0.5,0.75")
    p.add_argument("--seeds", default="41,42,43")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--window", type=int, default=512)
    p.add_argument("--block-size", type=int, default=2048)
    p.add_argument("--dense", action="store_true")
    p.add_argument("--no-ckpt", action="store_true")
    p.add_argument("--latent-clamp", type=float, default=None)
    p.add_argument("--latent-zscore", action="store_true")
    p.add_argument("--decode-topk", type=int, default=None)
    p.add_argument("--prefill-chunk", type=int, default=None)
    p.add_argument(
        "--decode-quest",
        type=int,
        default=None,
        help="quest baseline: select this many KV pages per kv head at decode",
    )
    p.add_argument(
        "--task",
        choices=["niah", "vt", "fwe", "qa", "cwe"],
        default="niah",
        help="RULER subtask: niah (single needle), vt, fwe, qa, cwe",
    )
    args = p.parse_args()
    run_ruler(
        args.model_path,
        args.ckpt,
        args.out,
        [int(x) for x in args.lengths.split(",")],
        [float(x) for x in args.depths.split(",")],
        [int(x) for x in args.seeds.split(",")],
        args.max_new_tokens,
        window=args.window,
        block_size=args.block_size,
        dense=args.dense,
        no_ckpt=args.no_ckpt,
        latent_clamp=args.latent_clamp,
        decode_topk=args.decode_topk,
        prefill_chunk=args.prefill_chunk,
        latent_zscore=args.latent_zscore,
        decode_quest=args.decode_quest,
        task=args.task,
    )


if __name__ == "__main__":
    main()
