#!/usr/bin/env bash
# Detach-launcher for run_burst5.sh — keeps shell operators out of gcloud --command.
mkdir -p ~/box_out_burst5
cd ~/lethe || exit 1
nohup bash scratch/run_burst5.sh > ~/box_out_burst5/burst5.log 2>&1 &
echo "launched pid=$!"
