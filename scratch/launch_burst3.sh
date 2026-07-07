#!/usr/bin/env bash
# Detach-launcher for run_burst3.sh — keeps shell operators out of gcloud --command.
mkdir -p ~/box_out_burst3
cd ~/lethe || exit 1
nohup bash scratch/run_burst3.sh > ~/box_out_burst3/burst3.log 2>&1 &
echo "launched pid=$!"
