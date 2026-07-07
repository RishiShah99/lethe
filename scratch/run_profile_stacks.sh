#!/usr/bin/env bash
# #50a — with_stack profile to pin the cw backward's gemv/glue to Python call sites.
#   nohup bash scratch/run_profile_stacks.sh > ~/profile_stacks.log 2>&1 & echo PID=$!
#   grep -E 'EXIT_|% of full|call site|gemv|wrote' ~/profile_stacks.log
set -u
cd ~/lethe
mkdir -p results
PYTHONPATH=src:. ~/cuteenv/bin/python scratch/gdn2_profile_stacks.py
echo "EXIT_PROFILE_STACKS=$?"
