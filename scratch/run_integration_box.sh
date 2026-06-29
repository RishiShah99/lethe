#!/usr/bin/env bash
# Self-healing box runner for the Phase-2 integration gate.
# Reuses ~/cuteenv (torch cu128 + nvidia-cutlass-dsl); builds it if absent.
# Runs scratch/gdn2_integration_box.py (native assembly vs oracle on B200).
# All shell operators live here so the fleet/ssh arg stays operator-free.
set +e
export PATH="$HOME/.local/bin:$PATH"

VENV="$HOME/cuteenv"
if [ ! -x "$VENV/bin/python" ]; then
  echo "===== BOOTSTRAP UV + VENV ====="
  command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
  uv venv "$VENV" --python 3.12 2>&1 | tail -4
  uv pip install --python "$VENV/bin/python" torch --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -4
  uv pip install --python "$VENV/bin/python" nvidia-cutlass-dsl numpy 2>&1 | tail -6
fi
PY="$VENV/bin/python"
echo "PY=$PY"; "$PY" --version

if [ -d /usr/local/cuda ]; then export CUDA_HOME=/usr/local/cuda; export PATH="$CUDA_HOME/bin:$PATH"; fi

echo ""
echo "===== GPU ====="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1 | head -2

echo ""
echo "===== VERSIONS ====="
"$PY" -c "import torch,cutlass; print('torch',torch.__version__,'cuda',torch.version.cuda); print('cutlass',getattr(cutlass,'__version__','?')); from cutlass.cute.nvgpu import tcgen05; print('tcgen05 ok')" 2>&1 | head -8

mkdir -p results
echo ""
echo "===== INTEGRATION GATE ====="
PYTHONPATH="$PWD/src:$PWD" "$PY" scratch/gdn2_integration_box.py --out results/gdn2_integration_box.json

echo ""
echo "############## INTEGRATION GATE DONE ##############"
