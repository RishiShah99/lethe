#!/usr/bin/env bash
# Detach-launcher for run_burst4.sh — keeps shell operators out of gcloud --command.
mkdir -p ~/box_out_burst4
cd ~/lethe || exit 1
nohup bash scratch/run_burst4.sh > ~/box_out_burst4/burst4.log 2>&1 &
echo "launched pid=$!"
