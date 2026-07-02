#!/usr/bin/env bash
# Burst-2 round B — targeted re-runs after the first-round desk fixes.
#   f2: graph re-bench with event-path timing (real graph numbers, all 4 shapes)
#   i2: Level-3 fused-kernel gates (bundle builder now in-process)
#   h2: Level-2 epilogue-glue gate (runtime call-args convention fixed)
#   j2: tiny-train native arm (enable_grad fix)
set -o pipefail
OUTDIR=~/box_out_burst2
mkdir -p "$OUTDIR"
PY=~/cuteenv/bin/python
PP=PYTHONPATH=src:.

_step() { echo ""; echo "================================================================"; echo "  STEP: $1"; echo "================================================================"; }
_rc()   { echo "  [$1] exit_code=$2"; }

_step "f2: graph re-bench (event-path timing)"
eval "$PP" "$PY" scratch/gdn2_graph_bench.py \
    --out "$OUTDIR/gdn2_graph_bench_v3.json" \
    > "$OUTDIR/gdn2_graph_bench_v3.log" 2>&1
_rc "f2" "$?"

_step "i2: k1_incb2_offsets (scalar, then cw)"
for MODE in scalar cw; do
    eval "$PP" "$PY" scratch/k1_incb2_offsets.py --mode "$MODE" \
        --out "$OUTDIR/k1_incb2_v2_${MODE}_r2.json" \
        > "$OUTDIR/k1_incb2_v2_${MODE}_r2.log" 2>&1
    _rc "i2(${MODE})" "$?"
done

_step "h2: k1_l2_epilogue --mode cw"
eval "$PP" "$PY" scratch/k1_l2_epilogue.py --mode cw \
    --out "$OUTDIR/k1_l2_epilogue_r2.json" \
    > "$OUTDIR/k1_l2_epilogue_r2.log" 2>&1
_rc "h2" "$?"

_step "j2: gdn2_tiny_train --arm native"
eval "$PP" "$PY" scratch/gdn2_tiny_train.py --arm native --steps 300 \
    --out "$OUTDIR/gdn2_tiny_train_native_r2.json" \
    > "$OUTDIR/gdn2_tiny_train_native_r2.log" 2>&1
_rc "j2" "$?"

echo ""
echo "ROUND B COMPLETE"
ls -la "$OUTDIR/" | tail -12
