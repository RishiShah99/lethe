#!/usr/bin/env bash
# Runner for the single-tile tcgen05 GEMM smoke. Reuses the cuteenv from the DSL
# de-risk (torch + nvidia-cutlass-dsl already installed). Operator-free fleet arg.
set +e
export PATH="$HOME/.local/bin:$PATH"
VENV="$HOME/cuteenv"
PY="$VENV/bin/python"
if [ ! -x "$PY" ]; then
  echo "cuteenv missing; run scratch/run_cute_smoke.sh first"; exit 1
fi
if [ -d /usr/local/cuda ]; then
  export CUDA_HOME=/usr/local/cuda
  export PATH="$CUDA_HOME/bin:$PATH"
fi
echo "===== GPU ====="
nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1
echo ""
echo "===== RUN TCGEN05 GEMM SMOKE ====="
PYTHONPATH="$PWD/src:$PWD" "$PY" scratch/tcgen05_gemm_smoke.py
echo ""
echo "############## TCGEN05 SMOKE DONE ##############"
