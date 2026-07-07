#!/bin/bash
# Wait for the detached backward-rigor run to finish, persist its result to the
# boot disk, then power the VM off (TERMINATED — disk persists, billing stops).
# Launched detached so the laptop can disconnect; bounded (~120 min) so a hung
# run still shuts the box down. Completion is signalled by the driver's final
# SCAN_MODE_BWD_RIGOR_JSON marker (the log is block-buffered and flushes when
# the run exits), with a no-process fallback.
cd "$HOME/lethe" || exit 1
LOG=rigor_bwd.log

for _ in $(seq 1 240); do  # up to ~120 min
    if grep -q 'SCAN_MODE_BWD_RIGOR_JSON' "$LOG" 2>/dev/null; then break; fi
    if ! pgrep -f 'python scratch/scan_mode_backward_rigor' >/dev/null 2>&1; then
        sleep 20  # process gone — give the log a moment to flush, then stop
        break
    fi
    sleep 30
done

mkdir -p results
if grep -q 'SCAN_MODE_BWD_RIGOR_JSON' "$LOG" 2>/dev/null; then
    grep -o 'SCAN_MODE_BWD_RIGOR_JSON .*' "$LOG" \
        | tail -1 | sed 's/^SCAN_MODE_BWD_RIGOR_JSON //' \
        > results/scan_mode_backward_rigor.json
    echo "RIGOR_DONE_OK $(date -u)" >> "$LOG"
else
    echo "RIGOR_DONE_NO_JSON $(date -u)" >> "$LOG"
fi
sync
sudo shutdown -h now
