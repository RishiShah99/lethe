#!/usr/bin/env bash
# Generic detached launcher for the GPU box. Usage:
#   fleet run "bash scratch/detach.sh <command...>"
# Runs <command> from the repo root, detached from SSH (survives disconnect),
# output to train.log so `fleet logs` can follow it.
cd "$(dirname "$0")/.."
touch train.log
nohup sh -c "$*" > train.log 2>&1 &
echo "LAUNCHED pid $!"
