#!/usr/bin/env bash
# Detach-launcher for run_verifier_p1.sh — keeps shell operators out of gcloud --command.
mkdir -p ~/box_out_verifier
cd ~/lethe || exit 1
nohup bash scratch/run_verifier_p1.sh > ~/box_out_verifier/p1.log 2>&1 &
echo "launched pid=$!"
