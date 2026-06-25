#!/bin/bash
# Launch a DDP PTB-XL training run detached -> <CKPT>.log, with the #19
# regularization knobs. Args: CKPT_DIR STEPS EXTRA_ARGS...
# EXTRA_ARGS is passed through verbatim (e.g. "--config b1 --dropout 0.2
# --augment --lr-decay"). No spaces inside a single knob value (HANDOFF caveat).
cd "$HOME/flash-mamba-rl" || exit 1
export PATH=$HOME/.local/bin:$PATH
CKPT="${1:-ptbxl_reg}"
STEPS="${2:-4000}"
shift 2 2>/dev/null || true
nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run --no-sync torchrun --standalone --nproc_per_node=8 scratch/ptbxl_train.py \
    --data-root "$HOME/data/ptbxl" --steps "$STEPS" --batch-size 8 \
    --eval-every 250 --save-every 500 --log-every 25 --checkpoint-dir "$CKPT" "$@" \
    > "${CKPT}.log" 2>&1 &
echo "LAUNCHED ptbxl pid=$! ckpt=$CKPT steps=$STEPS extra=$*"
