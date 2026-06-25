#!/bin/bash
# #15: regenerate the SFT warm-start adapter (eager targets), then run the
# six-level curriculum from it with the fixed token budgets (4096 fwd /
# 12288 bwd are the phase_e_run.py defaults). Detached + self-shutdown on
# completion so the box powers off (TERMINATED, disk persists, billing stops).
# Re-runnable: SFT skips if the adapter already exists; the curriculum
# --resume picks up its committed level state after a spot preemption.
# Args: STEPS_SFT CKPT_SFT CKPT_CUR
cd "$HOME/flash-mamba-rl" || exit 1
export PATH=$HOME/.local/bin:$PATH
STEPS_SFT="${1:-240}"
CKPT_SFT="${2:-sft_out}"
CKPT_CUR="${3:-phase_e_out}"
GPUS=0,1,2,3,4,5,6,7

run_done() { sync; sudo shutdown -h now; }

# 1) SFT regen (eager variants) — fast, no scoring. Skip if already present.
ADAPTER=$(ls -d "$CKPT_SFT"/adapter_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1)
if [ -z "$ADAPTER" ]; then
    env CUDA_VISIBLE_DEVICES=$GPUS uv run --no-sync python scratch/sft_run.py \
        --variants eager --steps "$STEPS_SFT" --ckpt-dir "$CKPT_SFT" > sft_regen.log 2>&1
    ADAPTER=$(ls -d "$CKPT_SFT"/adapter_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1)
fi
echo "SFT adapter=$ADAPTER"
if [ -z "$ADAPTER" ]; then echo "SFT_FAILED_NO_ADAPTER"; run_done; fi

# 2) curriculum from the warm-start adapter (token budgets fixed in #15).
env CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run --no-sync python scratch/phase_e_run.py --mode curriculum \
    --init-adapter "$ADAPTER" --ckpt-dir "$CKPT_CUR" --resume > curriculum.log 2>&1
echo "CURRICULUM_DONE"
run_done
