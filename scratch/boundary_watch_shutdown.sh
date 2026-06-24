#!/bin/bash
# Wait for the detached scan_mode boundary sweep to finish, then power the VM off
# (TERMINATED -- disk persists, billing stops). Launched detached so the laptop
# can disconnect; bounded so a hung run still shuts the box down. The driver
# rewrites results/scan_mode_boundary.json after every shape, so no marshalling
# is needed here -- completion is the driver's SCAN_MODE_BOUNDARY_DONE marker,
# with a no-process fallback.
cd "$HOME/flash-mamba-rl" || exit 1
LOG=boundary.log

for _ in $(seq 1 1440); do  # up to ~12 h (bound only; true signal is DONE / process-gone)
    if grep -q 'SCAN_MODE_BOUNDARY_DONE' "$LOG" 2>/dev/null; then break; fi
    if ! pgrep -f 'python scratch/scan_mode_boundary' >/dev/null 2>&1; then
        sleep 20  # process gone -- give the log a moment to flush, then stop
        break
    fi
    sleep 30
done

if grep -q 'SCAN_MODE_BOUNDARY_DONE' "$LOG" 2>/dev/null; then
    echo "BOUNDARY_DONE_OK $(date -u)" >> "$LOG"
else
    echo "BOUNDARY_DONE_NO_MARKER $(date -u)" >> "$LOG"
fi
sync
sudo shutdown -h now
