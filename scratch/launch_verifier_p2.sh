#!/usr/bin/env bash
# Detach-launcher for run_verifier_p2.sh (T2 + T4 Sakana).
mkdir -p ~/box_out_verifier
cd ~/lethe || exit 1
nohup bash scratch/run_verifier_p2.sh > ~/box_out_verifier/p2.log 2>&1 &
echo "launched pid=$!"
