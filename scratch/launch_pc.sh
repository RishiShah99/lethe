#!/usr/bin/env bash
# Detached positive-control run (T2) from the fresh ~/lethe deploy.
mkdir -p ~/box_out_verifier
cd ~/lethe || exit 1
export PYTHONPATH=src:.
export TORCH_CUDA_ARCH_LIST=10.0
rm -f ~/box_out_verifier/PC_DONE
nohup bash -c '~/cuteenv/bin/python scratch/positive_control.py --device cuda --json ~/box_out_verifier/positive_control.json > ~/box_out_verifier/pc.log 2>&1; echo "exit=$?" > ~/box_out_verifier/PC_DONE' &
echo "launched pc pid=$!"
