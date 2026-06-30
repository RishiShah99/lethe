#!/usr/bin/env bash
set -u
cd ~/flash-mamba-rl
mkdir -p results
PYTHONPATH=src:. ~/cuteenv/bin/python scratch/gdn2_graph_bench.py --out results/gdn2_graph_bench.json
echo "EXIT_GRAPH_BENCH=$?"
