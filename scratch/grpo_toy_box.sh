#!/usr/bin/env bash
# Box-side GRPO toy validation: sync deps (peft is new in the rl extra),
# then launch the elementwise-SiLU run detached. Poll grpo_toy_out/metrics.jsonl.
set -e
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

# --inexact: never prune the provisioned-but-unlocked GPU stack
# (mamba_ssm / causal_conv1d live outside pyproject by design).
uv sync --inexact --extra gpu --extra rl --group dev 2>&1 | tail -3

CUDA_VISIBLE_DEVICES=0 nohup uv run python scratch/grpo_toy_run.py \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --steps 40 --k 8 --resume \
    --ckpt-dir grpo_toy_out \
    > train.log 2>&1 &
echo "LAUNCHED pid $!"
