#!/usr/bin/env bash
# Self-healing box runner for the Phase-1 GDN-2 Hopper KILL-GATE.
# Bootstrap uv -> venv -> torch(cu128) + flash-linear-attention (+ the deps the
# flash_mamba_rl import chain needs) -> run scratch/gdn2_hopper_xcheck.py.
# All shell operators live here so the fleet-train arg stays operator-free.
set +e
export PATH="$HOME/.local/bin:$PATH"

echo "===== BOOTSTRAP UV ====="
if ! command -v uv >/dev/null 2>&1; then
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh -o /tmp/uvinstall.sh
  else
    wget -qO /tmp/uvinstall.sh https://astral.sh/uv/install.sh
  fi
  sh /tmp/uvinstall.sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version || { echo "UV BOOTSTRAP FAILED"; exit 1; }

VENV="$HOME/gdnenv"
if [ ! -x "$VENV/bin/python" ]; then
  echo "===== CREATE VENV (py3.12) ====="
  uv venv "$VENV" --python 3.12 2>&1 | tail -8
fi
PY="$VENV/bin/python"
echo "PY=$PY"; "$PY" --version

echo "===== INSTALL TORCH (cu128, Hopper/Blackwell) ====="
uv pip install --python "$PY" torch --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -6

echo "===== INSTALL FLA (pulls triton, einops, transformers) ====="
uv pip install --python "$PY" flash-linear-attention 2>&1 | tail -10

echo "===== INSTALL flash_mamba_rl import-chain deps ====="
uv pip install --python "$PY" numpy einops psutil rich 2>&1 | tail -6

# CUDA toolkit (common-cu image) — harmless if absent; naive path doesn't need it.
if [ -d /usr/local/cuda ]; then
  export CUDA_HOME=/usr/local/cuda
  export PATH="$CUDA_HOME/bin:$PATH"
fi

# Triton JIT-compiles a cuda_utils.c that needs Python.h. The GO gate (scalar
# reduction + determinism) is pure-torch and does NOT need this; it only unlocks
# the bonus fla-Triton-kernel cross-check. Best-effort.
echo "===== ENSURE PYTHON HEADERS (triton JIT, bonus check only) ====="
if [ ! -f /usr/include/python3.12/Python.h ]; then
  sudo apt-get update -y 2>&1 | tail -2
  sudo apt-get install -y python3-dev python3.12-dev 2>&1 | tail -3
fi
ls -l /usr/include/python3.12/Python.h 2>&1 || echo "no system Python.h (bonus check will skip)"

echo ""
echo "===== GPU ====="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1 | head -2

echo ""
echo "===== IMPORT SANITY ====="
PYTHONPATH="$PWD/src:$PWD" "$PY" -c "import torch, fla, flash_mamba_rl.kernels.references.gdn_backward as m; print('import OK', torch.__version__, torch.cuda.is_available())"

echo ""
echo "===== RUN HOPPER KILL-GATE ====="
PYTHONPATH="$PWD/src:$PWD" "$PY" scratch/gdn2_hopper_xcheck.py

echo ""
echo "############## XCHECK DONE ##############"
