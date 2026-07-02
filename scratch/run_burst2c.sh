#!/usr/bin/env bash
# Burst-2 round C — final re-runs: L3 gates (Path fix), L2 gate (fence order), native train (fp32).
set -o pipefail
OUTDIR=~/box_out_burst2
mkdir -p "$OUTDIR"
PY=~/cuteenv/bin/python
PP=PYTHONPATH=src:.

_rc() { echo "  [$1] exit_code=$2"; }

for MODE in scalar cw; do
    eval "$PP" "$PY" scratch/k1_incb2_offsets.py --mode "$MODE" \
        --out "$OUTDIR/k1_incb2_v2_${MODE}_r3.json" \
        > "$OUTDIR/k1_incb2_v2_${MODE}_r3.log" 2>&1
    _rc "i3(${MODE})" "$?"
done

eval "$PP" "$PY" scratch/k1_l2_epilogue.py --mode cw \
    --out "$OUTDIR/k1_l2_epilogue_r3.json" \
    > "$OUTDIR/k1_l2_epilogue_r3.log" 2>&1
_rc "h3" "$?"

eval "$PP" "$PY" scratch/gdn2_tiny_train.py --arm native --steps 300 \
    --out "$OUTDIR/gdn2_tiny_train_native_r3.json" \
    > "$OUTDIR/gdn2_tiny_train_native_r3.log" 2>&1
_rc "j3(native)" "$?"

eval "$PP" "$PY" scratch/gdn2_tiny_train.py --arm eager --steps 300 \
    --out "$OUTDIR/gdn2_tiny_train_eager_r3.json" \
    > "$OUTDIR/gdn2_tiny_train_eager_r3.log" 2>&1
_rc "j3(eager)" "$?"

echo "ROUND C COMPLETE"
