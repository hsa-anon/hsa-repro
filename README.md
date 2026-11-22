# Anonymized Artifact: Three-Tier Hierarchical Sparse Attention

Reproducibility artifact for the submission "Bounded prefill width is free,
query-dependent decode reach is not: a boundary map for three-tier attention
on consumer devices."

## What is here

```
src/hsa/           model + training + evaluation code
  attention/       three-tier attention kernels (torch, chunked prefill)
  train/           distillation trainer (peeled per-layer backward), block-mass
                   distillation, needle injection
  eval/            RULER NIAH harness, tier decode cache (top-k retrieval),
                   Quest baseline cache (kernel-faithful port)
  index/           serving-path retrieval index (Ward bucketing, incremental)
tests/             offline unit tests (no GPU required for most)
scripts/           benchmark entry points (prompts, runner, log parsing)
results/           run logs + result JSONs backing the paper's tables
results/m2_ckpt/train_log.json   checkpoint training metadata (weights excluded)
```

## Artifact map

| Paper claim | Artifact |
|---|---|
| Structural limit: static tiers 0/27 across 9 checkpoints (RULER 4K) | `results/ckpt_sweep.json`, `results/static8000_d*.json` |
| Untrained window-boundary control: 0/3 outside vs 3/3 inside window | `results/static_untr_d*.json` |
| Decode mass diagnostics | `results/diag_mass_20261109.txt`, `diag_mass_sweep.json`, `diag_tf_probe_20261109.txt` |
| Paired top-k retrieval ablation: 3/3 vs 0/3 | `results/ablation_topk_d*.json` |
| Dense floor (950DT, 4K/32K/128K/256K; OOM at 512K/1M) | `results/dt152_20261122/` (`t1.log`, `summary.md`), `results/ruler_dense_npu*.json`, `dense_ruler_*.json` |
| Official Quest port: parity 1.8e-07, full-budget = dense, 3/3 at 25% pages | `results/quest_baseline_2026-11-21/`, `results/quest_baseline_2026-11-21.md`, `src/hsa/eval/quest_cache.py`, `tests/test_quest_cache.py` |
| Loss-retrieval decoupling (4 distillation variants) | `results/p2_32k/` |
| Phase 2b supervision routes (needle-weighted CE, block-mass) | `results/p2b_stageA/`, `results/p2b_stageB/` |
| Serving-path sweep (4K/32K/131K, 9/9) | `results/ruler_*.log` |
| 1M prefill systems run (21.0 tok/s) | telemetry values cited in `results/dt152_20261122/summary.md`; raw 1M telemetry logs not redistributable at this size, available on request |

## Setup

```bash
pip install -e .
pytest tests/            # offline tests
```

## Hardware requirements for reruns

- Eval (4K-131K RULER): 27B bf16 hybrid model, >= 96 GB unified memory
  (Apple Silicon) or a >= 91 GB HBM accelerator; model weights are downloaded
  separately and are not included.
- 1M prefill: 128 GB Apple Silicon (M-series), ~14.5 hours per run.
- Training: single >= 91 GB accelerator card, dockerized torch_npu stack.

## Determinism

The 1M prefill runs are byte-identical across independent reruns (verified
v5/v7). Evaluation uses greedy decoding; identical prompts reproduce scores
exactly on the same backend.

## Excluded

- Model weights (27B bf16, ~50 GB) and distilled checkpoint weights (~450 MB
  per checkpoint); checkpoint training metadata is included.
- Raw 1M telemetry logs (multi-GB); summarized in `results/dt152_20261122/`.
