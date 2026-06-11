#!/usr/bin/env bash
# C5 post-fix validation: gate rerun + on-device floors + parity measurement.
# Box usage: fleet run "bash scratch/detach.sh bash scratch/c5_validate.sh"
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

echo "=== C5 validate: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
uv run pytest tests/test_kernels_gpu.py -m gpu -q -k C5 -rA
status=$?
echo "=== C5 gate rerun exit: $status ==="
echo "=== c5_b200_floor ==="
uv run python scratch/c5_b200_floor.py
echo "=== c5_parity_measure ==="
uv run python scratch/c5_parity_measure.py
echo "=== C5 validate done ==="
exit $status
