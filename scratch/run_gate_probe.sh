#!/usr/bin/env bash
# Self-healing box runner: bootstrap uv -> venv -> torch(cu128)+fla -> probe in
# BOTH fla modes. Single command in the fleet-train arg; all operators live here.
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

echo "===== INSTALL TORCH (cu128, Blackwell) ====="
uv pip install --python "$PY" torch --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -8

echo "===== INSTALL FLA (pulls triton) ====="
uv pip install --python "$PY" flash-linear-attention 2>&1 | tail -12

echo ""
echo "############## MODE: DEFAULT (TMA on) ##############"
"$PY" scratch/gdn_gate_probe.py

echo ""
echo "############## MODE: FLA_USE_TMA=0 ##############"
FLA_USE_TMA=0 "$PY" scratch/gdn_gate_probe.py

echo ""
echo "############## ALL MODES DONE ##############"
