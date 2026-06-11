#!/usr/bin/env bash
# Launch one audit shard per GPU, detached. Usage:
#   fleet run "bash scratch/audit_box.sh"
# Poll:  fleet run "tail -3 audit_out/shard*.log"
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
mkdir -p audit_out
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i nohup uv run python scratch/audit_run.py \
    scratch/audit_manifest_drkernel.jsonl.gz audit_out/results_shard${i}.jsonl \
    --device cuda --shard "$i" --num-shards 8 \
    > audit_out/shard${i}.log 2>&1 &
  echo "shard $i pid $!"
done
echo LAUNCHED
