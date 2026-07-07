#!/usr/bin/env bash
# #47 capture-safety probe — run detached on the B200 box, then poll ~/graph_probe.log.
#   nohup bash scratch/run_graph_probe.sh > ~/graph_probe.log 2>&1 & echo PID=$!
#   grep -E 'GO=|EXIT_|capture_ok|replay_worst' ~/graph_probe.log
set -u
cd ~/lethe
PYTHONPATH=src:. ~/cuteenv/bin/python scratch/graph_probe.py
echo "EXIT_GRAPH_PROBE=$?"
