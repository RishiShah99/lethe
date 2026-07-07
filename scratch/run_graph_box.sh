#!/usr/bin/env bash
# #48 — graph the whole cw backward: gate + timing on B200. Run detached, poll ~/graph_box.log.
#   nohup bash scratch/run_graph_box.sh > ~/graph_box.log 2>&1 & echo PID=$!
#   grep -E 'GO=|EXIT_|worst|speedup|capture' ~/graph_box.log
set -u
cd ~/lethe
mkdir -p results
PYTHONPATH=src:. ~/cuteenv/bin/python scratch/gdn2_graph_box.py --out results/gdn2_graph_box.json
echo "EXIT_GRAPH_BOX=$?"
