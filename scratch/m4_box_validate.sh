#!/usr/bin/env bash
# M4 landing validation on B200: the seeded contract gates must still pass the
# honest MIMO kernel on every view and reject the fp16-state cheat on grad_dt.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

echo ">> C3 GPU gate tests (honest kernel through the seeded run_all_gates)"
uv run pytest tests/test_kernels_gpu.py -k C3 -q 2>&1 | tail -8

echo ""
echo ">> M4 seed cheat-discrimination probe (honest all-pass + cheat fails grad_dt)"
uv run python scratch/m4_seed_validate.py
echo ">> m4_box_validate complete"
