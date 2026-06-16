#!/usr/bin/env bash
# Post-review-fix smoke: 4 fresh steps through the fixed hot path
# (temperature-scaled log-probs, termination-aware EOS, exact-equality
# degenerate guard, commit-point checkpoints).
set -e
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
CUDA_VISIBLE_DEVICES=0 nohup uv run python scratch/grpo_toy_run.py \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --steps 4 --k 8 \
    --ckpt-dir grpo_smoke_out \
    > smoke.log 2>&1 &
echo "LAUNCHED pid $!"
