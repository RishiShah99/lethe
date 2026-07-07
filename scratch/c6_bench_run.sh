#!/usr/bin/env bash
# C6 parity probe + full bench. Box usage:
# fleet run "bash scratch/detach.sh bash scratch/c6_bench_run.sh"
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

echo "=== c6_parity_measure: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
uv run python scratch/c6_parity_measure.py
echo "=== c6 bench ==="
uv run python -m lethe.bench.c6_fused_block_backward --out "$HOME/out/c6_bench.json"
echo "=== c6 bench done ==="
