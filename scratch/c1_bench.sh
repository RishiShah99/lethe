#!/usr/bin/env bash
# C1 benchmark sweep -> ~/out/c1_bench.json (fleet pull brings it home).
# Box usage: fleet run "bash scratch/detach.sh bash scratch/c1_bench.sh"
# Quick smoke: fleet run "bash scratch/detach.sh bash scratch/c1_bench.sh --quick"
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$HOME/out"
echo "=== C1 bench: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
uv run python -m flash_mamba_rl.bench.c1_forward_chunked_scan --out "$HOME/out/c1_bench.json" "$@"
status=$?
echo "=== C1 bench exit: $status ==="
exit $status
