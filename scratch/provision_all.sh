#!/usr/bin/env bash
# Full fresh-box provisioning for the PTB-XL training track (us-east1 B200).
# Self-contained: uv + env (gpu/rl/medical) + CUDA 13 toolkit + sanity.
# Launch detached via scratch/run_provision.sh -> provision.log.
set -euo pipefail
cd "$(dirname "$0")/.."

echo ">> system build deps (python3.12-dev: Triton JIT links cuda_utils against Python.h)"
sudo apt-get install -y -q python3.12-dev 2>&1 | tail -1

if ! command -v uv >/dev/null 2>&1; then
  echo ">> installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

echo ">> uv sync (gpu + rl + medical, dev group)"
uv sync --extra gpu --extra rl --extra medical

echo ">> install cuda 13 toolkit"
bash scratch/install_cuda13.sh

echo ">> torch / cuda / triton sanity"
uv run python - <<'PY'
import torch, triton
print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| avail", torch.cuda.is_available(), "| devices", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device0", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print("triton", triton.__version__)
PY

echo ">> PROVISION_ALL_COMPLETE"
