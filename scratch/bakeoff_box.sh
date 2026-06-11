#!/usr/bin/env bash
# Phase D bakeoff: sequential single-shot eval of coder models on the C1 spec.
# GPU-frugal: one model at a time (weights + scoring share the visible GPUs).
# Usage: fleet run "bash scratch/detach.sh bash scratch/bakeoff_box.sh"
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
mkdir -p bakeoff_out

run_model() {
  local model="$1" tag="$2"
  echo "=== bakeoff: $model $(date -u +%H:%M:%SZ) ==="
  CUDA_VISIBLE_DEVICES=0,1 uv run python scratch/bakeoff_run.py \
    --model "$model" --n 16 --out "bakeoff_out/${tag}.jsonl" --device cuda
  echo "=== done: $model exit $? ==="
}

run_model "Qwen/Qwen2.5-Coder-7B-Instruct" qwen25_coder_7b
run_model "Qwen/Qwen2.5-Coder-14B-Instruct" qwen25_coder_14b
run_model "Qwen/Qwen2.5-Coder-32B-Instruct" qwen25_coder_32b
run_model "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct" deepseek_coder_v2_lite
echo "BAKEOFF COMPLETE"
