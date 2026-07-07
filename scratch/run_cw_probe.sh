#!/usr/bin/env bash
set -u
cd ~/lethe
echo "===== L2=0 ====="
CUDA_LAUNCH_BLOCKING=1 PYTHONPATH=src:. ~/cuteenv/bin/python scratch/graph_cw_probe.py --l2 0; echo "EXIT_L2_0=$?"
echo "===== L2=1 ====="
CUDA_LAUNCH_BLOCKING=1 PYTHONPATH=src:. ~/cuteenv/bin/python scratch/graph_cw_probe.py --l2 1; echo "EXIT_L2_1=$?"
