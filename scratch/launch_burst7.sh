#!/usr/bin/env bash
# Detach-launcher for run_burst7.sh — keeps shell operators out of gcloud --command.
mkdir -p ~/box_out_burst7
cd ~/flash-mamba-rl || exit 1
nohup bash scratch/run_burst7.sh > ~/box_out_burst7/burst7.log 2>&1 &
echo "launched pid=$!"
