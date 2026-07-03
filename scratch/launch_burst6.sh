#!/usr/bin/env bash
# Detach-launcher for run_burst6.sh — keeps shell operators out of gcloud --command.
mkdir -p ~/box_out_burst6
cd ~/flash-mamba-rl || exit 1
nohup bash scratch/run_burst6.sh > ~/box_out_burst6/burst6.log 2>&1 &
echo "launched pid=$!"
