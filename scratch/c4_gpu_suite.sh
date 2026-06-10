#!/usr/bin/env bash
# C4 GPU validation: full kernel GPU suite (C1+C2+C3 regression + C4) + verifier.
# Box usage: fleet run "bash scratch/detach.sh bash scratch/c4_gpu_suite.sh"
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

echo "=== C4 GPU suite: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
uv run pytest tests/test_kernels_gpu.py tests/test_gpu_verifier.py -m gpu -q -rA
status=$?
echo "=== C4 GPU suite exit: $status ==="
exit $status
