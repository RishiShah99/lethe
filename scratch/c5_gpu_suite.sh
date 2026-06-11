#!/usr/bin/env bash
# C5 GPU validation: full kernel GPU suite (C1-C4 regression + C5) + verifier.
# Box usage: fleet run "bash scratch/detach.sh bash scratch/c5_gpu_suite.sh"
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

echo "=== C5 GPU suite: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
uv run pytest tests/test_kernels_gpu.py tests/test_gpu_verifier.py -m gpu -q -rA
status=$?
echo "=== C5 GPU suite exit: $status ==="
exit $status
