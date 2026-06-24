#!/bin/bash
# Launch the scan_mode boundary sweep detached -> boundary.log, plus the
# self-shutdown watcher. Bundled into a synced script because nested quotes do
# not survive cmd -> gcloud -> plink (HANDOFF caveat).
cd "$HOME/flash-mamba-rl" || exit 1
export PATH=$HOME/.local/bin:$PATH
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    uv run --no-sync python scratch/scan_mode_boundary.py \
    > boundary.log 2>&1 &
echo "LAUNCHED scan_mode_boundary pid=$!"
nohup bash scratch/boundary_watch_shutdown.sh > boundary_watch.log 2>&1 &
echo "LAUNCHED boundary_watch pid=$!"
