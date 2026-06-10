#!/usr/bin/env bash
# C1 GPU validation: kernel tests + full verifier GPU suite.
# Box usage: fleet run "bash scratch/detach.sh bash scratch/c1_gpu_suite.sh"
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

echo "=== C1 GPU suite: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
uv run pytest tests/test_kernels_gpu.py tests/test_gpu_verifier.py -m gpu -q -rA
status=$?
echo "=== C1 GPU suite exit: $status ==="
exit $status
