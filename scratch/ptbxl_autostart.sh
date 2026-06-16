#!/usr/bin/env bash
# Wait for the PTB-XL download to finish, then launch the 8-GPU DDP training run.
# Launch detached: nohup bash scratch/ptbxl_autostart.sh > ptbxl_train.log 2>&1 &
set -uo pipefail
cd ~/flash-mamba-rl
export PATH="$HOME/.local/bin:$PATH"
ROOT="$HOME/data/ptbxl"

echo ">> waiting for PTB-XL dataset (ptbxl_database.csv) ..."
for _ in $(seq 1 240); do
  [ -f "$ROOT/ptbxl_database.csv" ] && break
  sleep 15
done
if [ ! -f "$ROOT/ptbxl_database.csv" ]; then
  echo "DATASET_TIMEOUT — ptbxl_database.csv never appeared"
  exit 1
fi

echo ">> dataset ready; launching 8xB200 DDP training"
exec uv run --no-sync torchrun --standalone --nproc_per_node=8 scratch/ptbxl_train.py \
  --data-root "$ROOT" \
  --steps 20000 --batch-size 8 \
  --eval-every 500 --save-every 500 --log-every 10 \
  --resume
