#!/usr/bin/env bash
# Install the official mamba-ssm (+ causal-conv1d) into the box venv.
# Separate from provision.sh because it may compile CUDA extensions (slow)
# and the #904 reproduction depends on exactly which versions land here.
set -uo pipefail

cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

# Builds must see the CUDA 13.0 toolkit (matches torch +cu130).
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version | tail -1

echo ">> installing causal-conv1d + mamba-ssm (no build isolation; needs torch in env)"
uv pip install --no-build-isolation causal-conv1d 2>&1 | tail -5
uv pip install --no-build-isolation mamba-ssm 2>&1 | tail -5

echo ">> recording versions"
uv run python - <<'PY'
import torch, triton

print("torch", torch.__version__, "| cuda", torch.version.cuda)
print("triton", triton.__version__)
try:
    import mamba_ssm

    print("mamba_ssm", mamba_ssm.__version__)
except Exception as exc:  # broad on purpose: we want the failure mode recorded
    print("mamba_ssm import failed:", type(exc).__name__, exc)
try:
    import causal_conv1d

    print("causal_conv1d", causal_conv1d.__version__)
except Exception as exc:
    print("causal_conv1d import failed:", type(exc).__name__, exc)
PY
echo ">> install_mamba complete"
