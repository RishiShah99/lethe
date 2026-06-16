#!/usr/bin/env bash
# Detach provision_all.sh from the SSH session so it survives disconnect.
# Usage on box (via fleet run): bash scratch/run_provision.sh
cd "$(dirname "$0")/.."
nohup bash scratch/provision_all.sh > provision.log 2>&1 &
echo "LAUNCHED provision pid $!"
