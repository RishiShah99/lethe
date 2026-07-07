#!/usr/bin/env bash
# Detach-launcher for run_verifier_p3_sakana.sh (Sakana re-run with toolkit on PATH).
mkdir -p ~/box_out_verifier
cd ~/lethe || exit 1
nohup bash scratch/run_verifier_p3_sakana.sh > ~/box_out_verifier/p3.log 2>&1 &
echo "launched pid=$!"
