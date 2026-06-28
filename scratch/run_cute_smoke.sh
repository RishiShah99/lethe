#!/usr/bin/env bash
# Self-healing box runner for the CuTe DSL toolchain de-risk (Phase 2).
# Bootstrap uv -> venv -> torch(cu128) + nvidia-cutlass-dsl -> run scratch/cute_dsl_smoke.py.
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

VENV="$HOME/cuteenv"
if [ ! -x "$VENV/bin/python" ]; then
  echo "===== CREATE VENV (py3.12) ====="
  uv venv "$VENV" --python 3.12 2>&1 | tail -8
fi
PY="$VENV/bin/python"
echo "PY=$PY"; "$PY" --version

echo "===== INSTALL TORCH (cu128, Blackwell) ====="
uv pip install --python "$PY" torch --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -6

echo "===== INSTALL nvidia-cutlass-dsl (CuTe DSL) ====="
uv pip install --python "$PY" nvidia-cutlass-dsl 2>&1 | tail -12
uv pip install --python "$PY" numpy 2>&1 | tail -3

# CUDA toolkit (ptxas) — the DSL bundles its own, but expose the system one if present.
if [ -d /usr/local/cuda ]; then
  export CUDA_HOME=/usr/local/cuda
  export PATH="$CUDA_HOME/bin:$PATH"
fi

echo ""
echo "===== GPU ====="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1 | head -2

echo ""
echo "===== VERSIONS ====="
"$PY" -c "import cutlass; print('cutlass', getattr(cutlass,'__version__','?')); import cutlass.cute as cute; print('cute ok'); from cutlass.cute.nvgpu import tcgen05; print('tcgen05 ok')" 2>&1 | head -12

echo ""
echo "===== RUN CUTE DSL SMOKE ====="
PYTHONPATH="$PWD/src:$PWD" "$PY" scratch/cute_dsl_smoke.py

echo ""
echo "############## CUTE SMOKE DONE ##############"
