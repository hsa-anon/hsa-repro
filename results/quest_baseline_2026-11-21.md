# Quest page-bound external baseline - 2026-11-21 (node 10.8.82.152)

Model: qwen38-27b-bf16 + tier ckpt_8000 attached (16 full-attn layers, window=512 block=2048)
Protocol: ruler_tier --depths 0.75 --seeds 41,42,43 --max-new-tokens 512 --prefill-chunk 1024
Harness validated: dense control 3/3 hit with direct answers (results_g/dense_probe log; JSON write failed on missing dir, scores in log lines).

## 4096 (64 pages) - results_g/bisect + results_g/q*
| budget | P | hits |
|---|---|---|
| dense | - | 3/3 |
| 100% | 64 | 3/3 |
| 50% | 32 | 2/3 |
| 25% | 16 | 0/3 |
| 12.5% | 8 | 0/3 |
| 6.25% | 4 | 0/3 |
| 1.6% | 1 | 0/3 |

## 32768 (512 pages) - results_g/q4-q6 + results_g/bisect
| budget | P | hits |
|---|---|---|
| 50% | 256 | 1/3 |
| 25% | 128 | 0/3 |
| 12.5% | 64 | 0/3 |
| 6.25% | 32 | 0/3 |
| 3.1% | 16 | 0/3 |
| 0.8% | 4 | 0/3 |
| 0.2% | 1 | 0/3 |

Finding: raw q-dot-k Quest page selection needs >=~50% page budget at 4096; at 32768
retrieval collapses across all tested budgets. Answers at failing budgets drift into
think-style rambling that never emits the needle value (512-token cap).
Grid runs results_g/q0-q6 were the first-pass collection (P16/P4/P1 x 4096/32768, all 0/3);
bisect runs refine the curve. Container note: any run sharing a davinciN with a live
container silently falls back to CPU (torch.npu.is_available() False) - always
docker rm -f finished job containers before reusing a card.
