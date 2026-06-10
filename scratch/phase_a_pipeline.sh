#!/usr/bin/env bash
# Phase A on-box pipeline: GPU verifier tests -> mamba-ssm install -> env capture.
# Run detached:  fleet run "bash scratch/detach.sh bash scratch/phase_a_pipeline.sh"
# Poll:          fleet run "tail -20 train.log"
set -uo pipefail

cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

echo "=== [1/4] GPU verifier validation suite ==="
uv run pytest tests/test_gpu_verifier.py -v 2>&1
echo "=== [1/4] exit: $? ==="

echo "=== [2/4] CUDA 13.0 toolkit ==="
bash scratch/install_cuda13.sh 2>&1
echo "=== [2/4] exit: $? ==="

echo "=== [3/4] mamba-ssm install ==="
bash scratch/install_mamba.sh 2>&1
echo "=== [3/4] exit: $? ==="

echo "=== [4/4] environment capture ==="
uv run python scratch/capture_env.py 2>&1
echo "=== [4/4] exit: $? ==="

echo "=== PHASE A PIPELINE DONE ==="
