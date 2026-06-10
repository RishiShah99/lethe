#!/usr/bin/env bash
# Phase A continuation: wait out the dpkg lock, install CUDA 13 + mamba-ssm,
# re-run the GPU suite, capture env, attempt the #904 reproduction.
# Run detached:  fleet run "bash scratch/detach.sh bash scratch/phase_a_continue.sh"
set -uo pipefail

cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

echo "=== [1/5] wait for dpkg lock ==="
for _ in $(seq 1 60); do
  if sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; then
    sleep 10
  else
    break
  fi
done
echo "lock free"

echo "=== [2/5] CUDA 13.0 toolkit ==="
bash scratch/install_cuda13.sh 2>&1
echo "=== [2/5] exit: $? ==="

echo "=== [3/5] mamba-ssm install ==="
bash scratch/install_mamba.sh 2>&1
echo "=== [3/5] exit: $? ==="

echo "=== [4/5] GPU verifier suite (post-fixes) + env capture ==="
uv run pytest tests/test_gpu_verifier.py -q 2>&1 | tail -3
uv run python scratch/capture_env.py > /dev/null 2>&1
rc=$?
[ "$rc" -eq 0 ] && echo "env captured"
echo "=== [4/5] exit: $rc ==="

echo "=== [5/5] #904 reproduction (single GPU) ==="
CUDA_VISIBLE_DEVICES=0 uv run python scratch/repro_904.py 2>&1 | tail -25
echo "=== [5/5] exit: $? ==="

echo "=== PHASE A CONTINUE DONE ==="
