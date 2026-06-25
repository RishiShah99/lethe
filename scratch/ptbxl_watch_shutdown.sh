#!/bin/bash
# Wait for the detached PTB-XL training run to finish, then power the VM off.
# Detached so the laptop can disconnect. Bounded so a hung run still shuts down.
cd "$HOME/flash-mamba-rl" || exit 1
LOG="${1:-ptbxl_reg.log}"
for _ in $(seq 1 2880); do  # up to ~48 h at 60 s/tick (training is long)
    if ! pgrep -f 'scratch/ptbxl_train.py' >/dev/null 2>&1; then
        sleep 20
        break
    fi
    sleep 60
done
sync
echo "PTBXL_DONE $(date -u)" >> "$LOG"
sync
sudo shutdown -h now
