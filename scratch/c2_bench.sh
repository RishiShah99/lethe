#!/usr/bin/env bash
# C2 benchmark sweep -> ~/out/c2_bench.json (fleet pull brings it home).
# Box usage: fleet run "bash scratch/detach.sh bash scratch/c2_bench.sh"
# Quick smoke: fleet run "bash scratch/detach.sh bash scratch/c2_bench.sh --quick"
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$HOME/out"
echo "=== C2 bench: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
uv run python -m lethe.bench.c2_backward_selective_scan --out "$HOME/out/c2_bench.json" "$@"
status=$?
echo "=== C2 bench exit: $status ==="
exit $status
