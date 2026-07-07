#!/bin/bash
# Launch the E2.c config-RL scan_mode run detached -> e2_config.log.
# Args: LEVELS STEPS K CKPT_DIR [RESUME_FLAG]
# Bundled into a synced script because nested quotes do not survive
# cmd -> gcloud -> plink (HANDOFF caveat); --levels carries commas/colons.
cd "$HOME/lethe" || exit 1
export PATH=$HOME/.local/bin:$PATH
LEVELS="$1"
STEPS="${2:-40}"
K="${3:-16}"
CKPT="${4:-e2_config_out}"
RESUME="${5:-}"
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    uv run --no-sync python scratch/e2_config_rl.py \
    --levels "$LEVELS" --steps "$STEPS" --k "$K" --ckpt-dir "$CKPT" $RESUME \
    > "${CKPT}.log" 2>&1 &
echo "LAUNCHED e2_config_rl pid=$! ckpt=$CKPT steps=$STEPS k=$K"
