#!/usr/bin/env bash
# Box-side Phase E launch: trainer on GPU 0 (32B fits one B200 in bf16),
# scoring sandboxes pinned to GPUs 1-7 by the driver. Detached; poll with
# scratch/phase_e_poll.py or tail train.log.
#
# Pre-flight (the audit-burst uv sync pruned the provisioned stack):
#   bash scratch/install_mamba.sh   # before any official-comparator bench
set -e
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

# --inexact: never prune the provisioned-but-unlocked GPU stack
# (mamba_ssm / causal_conv1d live outside pyproject by design).
uv sync --inexact --extra gpu --extra rl --group dev 2>&1 | tail -3

CUDA_VISIBLE_DEVICES=0 nohup uv run python scratch/phase_e_run.py \
    --resume "$@" \
    >> train.log 2>&1 &
echo "LAUNCHED pid $!"
