#!/usr/bin/env bash
# scan_mode selector GPU regression: the public ops now route via _resolve_scan_mode
# at config=None, so the default-path kernel parity/gate tests must stay green.
# Always --no-sync (mamba_ssm/causal_conv1d live outside pyproject; a bare `uv run`
# would prune them and break the official-parity comparisons).
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

echo "=== selector GPU regression: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
if ! uv run --no-sync python -c "import mamba_ssm, causal_conv1d" 2>/dev/null; then
  echo "mamba_ssm/causal_conv1d missing -> reinstalling"
  bash scratch/install_mamba.sh
fi
uv run --no-sync python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

uv run --no-sync pytest tests/test_kernels_gpu.py tests/test_gpu_verifier.py -m gpu -q -rA
status=$?
echo "=== selector GPU regression exit: $status ==="
echo "SELECTOR_REGRESSION_DONE status=$status"
exit $status
