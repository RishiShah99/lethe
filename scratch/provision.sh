#!/usr/bin/env bash
# Idempotent GPU-box provisioning for lethe.
# Run on the box from the repo root:  bash scratch/provision.sh
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo ">> installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

echo ">> syncing python env (gpu + rl extras, dev group)"
uv sync --extra gpu --extra rl

echo ">> sanity: torch / cuda / triton"
uv run python - <<'PY'
import torch

print("torch", torch.__version__, "| built for cuda", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("devices:", torch.cuda.device_count())
    print("device 0:", torch.cuda.get_device_name(0),
          "| capability:", torch.cuda.get_device_capability(0))
import triton

print("triton", triton.__version__)
PY

echo ">> provision complete"
