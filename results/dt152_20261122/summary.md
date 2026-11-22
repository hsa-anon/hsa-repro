# 950DT (10.8.82.152) dense-floor measurements - 2026-11-22

Issue #10 T1/T3/T4/T5 execution record. Bilingual (EN primary, CN notes).
Artifacts: `t1/ t3/ t4/` (JSONs pulled from node), `t{332k,3131k,4,1}.log` (raw docker logs).
Node: 950DT 91.75GB HBM, container `vllm-ascend:v0.30.0-a5`, model qwen38-27b-bf16,
dense control path (`--no-ckpt --dense`), `--max-new-tokens 512` everywhere.

## Status table

| Task | Scope | Result | Verdict |
|---|---|---|---|
| T1 | dense 256K/512K/1M NIAH | 256K 1/1 hit (138s); 512K OOM; 1M OOM | measured 256K + OOM telemetry |
| T3 | dense vt/fwe/qa/cwe x {32K,131K} x seeds 41-43 | see table below | floor established, has cracks |
| T4 | niah 131K 3 depth x 3 repeats | 9/9 | K8 denominator fixed |
| T5 | official Quest baseline | ported algorithm-faithfully (b24e8bc), validated on node | done |

## T1: dense long-context ceiling (task 1)

- 256K: `t1/dense_ruler_262144.json` hit=True, 138s wall (prefill-chunk 8192).
- 512K: OOM with prefill-chunk 16384 / 8192 / 4096 / 2048 (OOM at 77-81GB allocated
  each time; allocation request 1.2-4GB fails with 0.6-3.9GB free).
- 1M: OOM with 16384 / 8192 / 4096 (identical signature).
- Conclusion: the failure is not chunk-size-bound (2048-token chunks still fail);
  attention backend carries total-length-proportional overhead. Dense ceiling on
  91.75GB with this harness = 256K tokens, measured. 512K/1M dense = OOM telemetry,
  which replaces the paper's extrapolated rows with measured evidence: the tier path
  runs 1M where dense cannot even allocate 512K.

中文:512K 稠密在四档 chunk 下全部 OOM,失败与 chunk 大小无关(2048 chunk 激活应仅
~2GB 仍差 4GB),判定为注意力后端存在随总长增长的固有开销。稠密实测上限 = 256K;
512K/1M 的 OOM 记录直接替换论文外推行,tier 在稠密连分配都做不到的长度上继续出数。

## T3: dense floor for vt/fwe/qa/cwe (task 3)

Dense (node) vs LSA serving (M5 Max, calibrated generators, same seeds):

| task | dense 32K | dense 131K | serving 32K | serving 131K |
|---|---|---|---|---|
| vt   | 2/3 | 1/3 | 0/3 | 0/3 |
| fwe  | 3/3 | 3/3 | 2/3 | 0/3 |
| qa   | 3/3 | 3/3 | 3/3 | 3/3 |
| cwe  | 2/3 | 3/3 | 3/3 | 0/3 |

### VT decode-budget probe (follow-up, same day)

512-token misses are **budget-bound, not structural**. Re-run at
`--max-tokens 2048` (`results/dt152_20261122/vt2048/` on the node; serving logs
`~/.cache-mlx/logs/vt-probe-*` + sweep `ruler-vt-...-2048`):

- Serving 32K: strict 0/3, **semantic 3/3** — the model resolves all 4 hops via
  iterative decode-time retrieval, then re-formats the value `448362-336` as
  arithmetic (`= 448026`), which strict substring scoring rejects.
- Serving 131K: strict 0/3, semantic 1/3 — in 2/3 cases the hop-1 line carrying the
  numeric value is never retrieved (the model reports "the text does not provide a
  final numerical value"), while pure-name hop lines are. A real retrieval gap:
  name-similar pages crowd out the numeric anchor page in latent q-dot-k selection.
- Dense at 2048: 2/3 at both lengths (JSONs `vt2048/` on the node) — the dense floor
  itself is not clean on VT; the task is hard for both paths.

Verdict for the paper: report VT at 512 (strict) with the budget-probe as analysis —
LSA iterative retrieval solves multi-hop at 32K given budget; the 131K residual is a
page-selection gap on numeric-anchor pages, not a budget artifact.

Reading:

1. QA 6/6 on both paths - LSA retrieval parity, strongest positive evidence.
2. VT: dense itself is NOT a clean floor (2/3 at 32K, 1/3 at 131K; model recites the
   variable chain instead of resolving it). LSA serving collapses to 0/3 - VT is a
   hard task for both paths and a clear LSA weakness (multi-hop aggregation).
3. fwe/cwe at 131K: dense 3/3 vs serving 0/3 - clean LSA gap (multi-page aggregation
   fails on the serving path at length).
4. cwe 32K inversion (dense 2/3, serving 3/3): serving goes through the chat template,
   node dense runs raw completion - path asymmetry, noted, not investigated further.
5. Dense misses show model-level failure modes independent of LSA (base-word collapse,
   chain recitation) - the floor has cracks and the paper should not claim
   "dense = solved" for vt/cwe.

## Generator calibration (blocking bug found and fixed)

Synthetic coded fillers (zelvarNNNNNN etc.) tokenize to ~8 tokens/word on the Qwen3
tokenizer vs ~1.3 for natural text. Word-count filler sizing overshot ~4x: a "32K"
VT case built 131K tokens; 131K built 524K tokens (past the 262K max position).
Fixed by build-time calibration (build, tokenize, rescale target, rebuild; within 5%):
serving-repo `scripts/ruler32k_bench.py` (25de998), hsa `src/hsa/eval/ruler_tier.py`
(5d7556c). All numbers above are post-calibration (true token lengths).

## FWE/CWE synthetic-token caveat (paper reporting)

The FWE/CWE haystacks use synthetic coded tokens (zelvarNNNNNN / myrvikNNN), not
natural words - cleaner than original RULER FWE (stop-word dominance removed) but
less natural. Observed failure mode on BOTH paths: the model merges numbered variants
into the base word ("zelvar", "myrvik") and counts base words - a tokenizer-level
attractor. Dense fwe/cwe misses are this mode, not harness bugs. Report the
construction and this failure mode explicitly.

中文:FWE/CWE 干草堆为合成编码 token(比 RULER 原版干净、无停用词支配,但欠自然);
两条路径都观察到模型把带编号变体归并为基词计数 - 论文需注明构造方式与该失效形态。

## T4: K8 denominator (task 4)

`t4/dense_ruler_131072.json`: 9/9 (3 depths x 3 repeats, seeds 41-49). Paper K8 row
refreshed; "1 repeat" footnote removed.

## T5: official Quest (task 5, optional)

Repo cloned on node (`/mnt/sfs_turbo/quest-official`). As-is blockers, evidenced:
1. Kernels are CUDA C++ (`kernels/include/{topk,prefill,decode}/*.cuh`, CMake CUDA
   build) - not runnable on Ascend NPU.
2. Container torch: `2.10.0+cpu, cuda=False, npu=True`.
3. Official Quest targets standard llama attention; qwen38-27b is a hybrid GDN arch
   the kernels do not support.

### Official Quest port (algorithm-faithful, approved follow-up)

Full source read of mit-han-lab/quest (models/QuestAttention.py, utils/controller.py,
utils/__init__.py, kernels decode_page.cuh / decode_attn.cuh / approx_attn.cu). Algorithm
extracted from the CUDA kernels, not just the paper:

- Metadata = per-page per-channel elementwise running MAX (k-slot) and MIN (v-slot)
  of keys, maintained incrementally per appended token (`decode_page.cuh` append
  kernels; vec_reduct max/min).
- Estimate score = `sum_d max(q_d * kmax_d, q_d * kmin_d)` per query head - the box
  upper bound of q.k over the page, no ReLU, no clamp, no scale
  (`decode_attn.cuh` compute_max_possible).
- Decode flow: pages > budget -> estimate over metadata (excluding the current page),
  topk of budget-1 pages per query head, sparse FlashInfer decode over the selection
  plus the always-attended current page (`approx_attn.cu` passes last_page_idx into
  the paged KV structure). Pages <= budget -> full attention.
- Layer skipping: first `skip_layer=2` layers run full attention, the rest sparse
  (`llama.py` sets an infinite page budget before layer 2).

Our `QuestLayerCache` (hsa/src/hsa/eval/quest_cache.py) diverged in two places and was
aligned (hsa commit b24e8bc):

1. Score was `sum_d ReLU(min(q*kmax, q*kmin))` (paper-style clipped bound); now the
   official `sum_d max(q*kmax, q*kmin)`.
2. Current page used to compete for budget; now excluded from selection and always
   attended (budget-1 candidates + current page).

Kernel semantics replaced by equivalent torch ops (masked einsum + topk + SDPA-style
softmax), NPU-compatible. Validation: brute-force parity test against the official
semantics (1.8e-07 max abs diff), full-budget decode == dense attention (1.7e-07),
8/8 unit tests (`tests/test_quest_cache.py`).

Node re-measurement with the aligned implementation (containers hsa-t5q / hsa-t5q2,
cards 5-6, niah depth 0.75, seeds 41-43; JSONs `t5q/`):

| config | 4K | 32K | 131K |
|---|---|---|---|
| dense floor | 3/3 | 3/3 | - (9/9 at d{.25,.5,.75} from T4) |
| quest full budget (p >= pages) | - | 3/3 | - |
| quest p=64 (4K) / p=128 (32K) / p=512 (131K) | 3/3 | 3/3 | 3/3 |

Validation chain: full-budget quest reproduces the dense floor exactly (3/3 at 32K) -
selection inactive, only overhead. Sparse budgets (25% of pages) match dense 3/3 on
single-needle retrieval at every length. The port is the paper's Quest baseline:
"algorithm verified line-by-line against mit-han-lab/quest; CUDA kernels replaced by
equivalent torch ops" replaces the earlier infeasibility note.

## Ops notes (node)

- Access: ProxyJump + SSH keys installed on both hops (password lockout on the jump
  host locks the root account for everyone - keys remove that failure mode entirely).
- argparse: `--ckpt` was `required=True` while `--no-ckpt` is a separate flag - dense
  runs crashed at argument parsing; fixed in 5d7556c.
- Container rules that held: `-w $BASE` mandatory; mkdir inside the container cmd;
  job output lives in `docker logs`, not `docker run -d` stdout.
