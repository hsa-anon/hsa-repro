#!/bin/bash
# W7 eval: NSA baseline on the ablation protocol (4K, depths 0.75+0.95,
# seeds 41-43, matched budget = 2 blocks x 32 keys + window 512).
# Chains after the W6 seed expansion (waits for its W6-DONE marker).
set -u
cd /Users/dahai/fusion/hsa
source /Users/dahai/fusion/.venv/bin/activate
MODEL=/Users/dahai/.fusion-mlx/models/qwen38-27b
LOG=/tmp/w7_nsa_eval.log

while ! /usr/bin/grep -q "W6-DONE\|W6-PROCESS-EXITED-ABNORMALY" /tmp/w6_run.log 2>/dev/null; do sleep 60; done
if /usr/bin/grep -q "W6-PROCESS-EXITED-ABNORMALY" /tmp/w6_run.log; then
  echo "W6 failed - aborting W7 chain" && exit 1
fi
# W6 uses one model load per python invocation; between invocations memory
# is released by process exit - no extra cleanup needed.
echo "=== nsa baseline, d075 ===" >> "$LOG"
python -m hsa.eval.ruler_tier --model-path "$MODEL" \
  --ckpt results/m2_ckpt/ckpt_8000.pt --out results/nsa_d075.json \
  --lengths 4096 --depths 0.75 --seeds 41,42,43 --decode-nsa 2 --max-new-tokens 512 >> "$LOG" 2>&1
echo "=== nsa baseline, d095 ===" >> "$LOG"
python -m hsa.eval.ruler_tier --model-path "$MODEL" \
  --ckpt results/m2_ckpt/ckpt_8000.pt --out results/nsa_d095.json \
  --lengths 4096 --depths 0.95 --seeds 41,42,43 --decode-nsa 2 --max-new-tokens 512 >> "$LOG" 2>&1
for f in results/nsa_d075.json results/nsa_d095.json; do
  echo "--- $f"; python -c "import json;d=json.load(open('$f'));print(d['summary'])"
done
echo "W7-DONE"
