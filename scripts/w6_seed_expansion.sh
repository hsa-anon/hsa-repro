#!/bin/bash
# W6 seed expansion: ablation cells from 3 seeds (41-43) to 6 (41-46).
# Runs only the NEW seeds 44,45,46; merge happens in the paper table.
# Waits for any running 1M/bench job, frees the fusion-mlx server, then runs
# the three decode paths (retrieval / static-trained / static-untrained) at
# depths 0.75+0.95 in one model load each.
set -u
cd /Users/dahai/fusion/hsa
source /Users/dahai/fusion/.venv/bin/activate
MODEL=/Users/dahai/.fusion-mlx/models/qwen38-27b
LOG=/tmp/w6_seed_expansion.log

echo "[1/4] waiting for harness-gain bench to exit..."
while pgrep -f "harness-gain/run.ts" >/dev/null; do sleep 60; done
echo "[2/4] bench exited; freeing fusion-mlx server memory..."
PID=$(lsof -nP -tiTCP:11434 -sTCP:LISTEN 2>/dev/null | head -1)
if [ -n "$PID" ]; then kill "$PID"; sleep 8; fi
PID=$(lsof -nP -tiTCP:11434 -sTCP:LISTEN 2>/dev/null | head -1)
[ -n "$PID" ] && kill -9 "$PID"
sleep 10
memory_pressure | head -1

echo "[3/4] running seed expansion (seeds 44,45,46)..."
echo "=== retrieval (topk-64), d075+d095 ===" >> "$LOG"
python -m hsa.eval.ruler_tier --model-path "$MODEL" \
  --ckpt results/m2_ckpt/ckpt_8000.pt --out results/ablation_topk_d075_s4446.json \
  --lengths 4096 --depths 0.75 --seeds 44,45,46 --decode-topk 64 --max-new-tokens 512 >> "$LOG" 2>&1
python -m hsa.eval.ruler_tier --model-path "$MODEL" \
  --ckpt results/m2_ckpt/ckpt_8000.pt --out results/ablation_topk_d095_s4446.json \
  --lengths 4096 --depths 0.95 --seeds 44,45,46 --decode-topk 64 --max-new-tokens 512 >> "$LOG" 2>&1
echo "=== static trained, d075+d095 ===" >> "$LOG"
python -m hsa.eval.ruler_tier --model-path "$MODEL" \
  --ckpt results/m2_ckpt/ckpt_8000.pt --out results/static8000_d075_s4446.json \
  --lengths 4096 --depths 0.75 --seeds 44,45,46 --max-new-tokens 512 >> "$LOG" 2>&1
python -m hsa.eval.ruler_tier --model-path "$MODEL" \
  --ckpt results/m2_ckpt/ckpt_8000.pt --out results/static8000_d095_s4446.json \
  --lengths 4096 --depths 0.95 --seeds 44,45,46 --max-new-tokens 512 >> "$LOG" 2>&1
echo "=== static untrained, d075+d095 ===" >> "$LOG"
python -m hsa.eval.ruler_tier --model-path "$MODEL" \
  --no-ckpt --out results/static_untr_d075_s4446.json \
  --lengths 4096 --depths 0.75 --seeds 44,45,46 --max-new-tokens 512 >> "$LOG" 2>&1
python -m hsa.eval.ruler_tier --model-path "$MODEL" \
  --no-ckpt --out results/static_untr_d095_s4446.json \
  --lengths 4096 --depths 0.95 --seeds 44,45,46 --max-new-tokens 512 >> "$LOG" 2>&1

echo "[4/4] summaries:"
for f in results/*_s4446.json; do echo "--- $f"; python -c "import json;d=json.load(open('$f'));print(d['summary'])"; done
echo "W6-DONE"
