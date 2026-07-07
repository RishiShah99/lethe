#!/bin/bash
# Wait for the detached E2.f edit-RL run to finish, then power the VM off
# (TERMINATED -> disk persists, billing stops). Launched detached so the laptop
# can disconnect. The run is one continuous python process across all levels,
# so "process gone" reliably means done or crashed. Bounded so a hung run still
# shuts the box down.
cd "$HOME/lethe" || exit 1
LOG="${1:-edit_rl_out.log}"

for _ in $(seq 1 1440); do  # up to ~24 h at 60 s/tick
    if ! pgrep -f 'scratch/edit_rl.py' >/dev/null 2>&1; then
        sleep 20  # process gone -- give the log a moment to flush, then stop
        break
    fi
    sleep 60
done

sync
if grep -q '\[final\]' "$LOG" 2>/dev/null; then
    echo "EDIT_RL_DONE_OK $(date -u)" >> "$LOG"
else
    echo "EDIT_RL_DONE_NO_FINAL $(date -u)" >> "$LOG"
fi
sync
sudo shutdown -h now
