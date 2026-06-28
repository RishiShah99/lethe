#!/usr/bin/env bash
# Self-healing box runner for the K#1 (B4 reverse-state scan) micro-gate.
# Reuses ~/cuteenv (torch cu128 + nvidia-cutlass-dsl), generates the bundle on-box
# from the verified chunkwise reference, then runs the kernel + comparison.
# All shell operators live here so the fleet-train arg stays operator-free.
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

NT="${NT:-1}"
BUN="k1_bundle_nt${NT}.pt"
echo ""
echo "===== GEN BUNDLE (NT=$NT) ====="
PYTHONPATH="$PWD/src" "$PY" scratch/gen_k1_bundle.py --nt "$NT" --out "$BUN" 2>&1 | tail -20

echo ""
echo "===== K#1 MICRO-GATE ====="
PYTHONPATH="$PWD/src:$PWD" "$PY" scratch/k1_microgate.py --bundle "$BUN"

echo ""
echo "############## K1 MICRO-GATE DONE ##############"
