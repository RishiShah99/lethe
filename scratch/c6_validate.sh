#!/usr/bin/env bash
# C6 post-fix validation: gate rerun + on-device floors + parity measurement.
# Box usage: fleet run "bash scratch/detach.sh bash scratch/c6_validate.sh"
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

echo "=== C6 validate: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
uv run pytest tests/test_kernels_gpu.py -m gpu -q -k C6 -rA
status=$?
echo "=== C6 gate rerun exit: $status ==="
echo "=== c6_cmp03_probe ==="
uv run python scratch/c6_cmp03_probe.py
echo "=== c6_b200_floor ==="
uv run python scratch/c6_b200_floor.py
echo "=== c6_parity_measure ==="
uv run python scratch/c6_parity_measure.py
echo "=== C6 validate done ==="
exit $status
