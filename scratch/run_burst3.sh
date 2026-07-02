#!/usr/bin/env bash
# Box-side burst-3 orchestration.  Runs from the repo root on the B200 box.
# Each step independent (|| true); JSON in ~/box_out_burst3/.
#
# Usage:
#   nohup bash scratch/run_burst3.sh > ~/box_out_burst3/burst3.log 2>&1 &
#
# Dependency order:
#   a. env capture + pristine .so md5 check (a1e3ba0a — the burst-2 control)
#   b. integration gate cw (default + --closed) through src dispatch — NOW WITH THE
#      LEVEL-2 K#1 DEFAULT (selector in gdn2_bwd_dhu_cw.run_k1_incB). If b FAILS,
#      every later step re-runs with FMR_DISABLE_L2=1 (the selector kill-switch).
#   c. tiny GDN-2 training, native arm — the drifted-regime NaN fix confirm
#      (masked-exponent decay_rel; desk assembly arm GO at span 270).
#   d. graph re-bench v4, event-path discipline (4 shapes, closed arm, vs fla).
#   e. Level-3 v3 gates (unrolled fused kernel, offset lifecycle): scalar nt=4,
#      cw nt=4, then nt=8 compile-scale probe. timeout-guarded (compile-time risk).
#   f. promoted-L2 module re-gate through the src import path (repro-path check).

set -o pipefail
OUTDIR=~/box_out_burst3
mkdir -p "$OUTDIR"
PY=~/cuteenv/bin/python
PP="PYTHONPATH=src:."
SO_GOOD_MD5=a1e3ba0a1e62227bccbea2aaf20cb6e8

_step() { echo ""; echo "================================================================"; echo "  STEP: $1"; echo "================================================================"; }
_rc()   { echo "  [$1] exit_code=$2"; }

_so_md5() {
    find ~/cuteenv/lib/python3.12/site-packages/nvidia_cutlass_dsl -name 'libcute_dsl_runtime.so' \
        -exec md5sum {} \; 2>/dev/null | head -1
}

# ── a. env capture + pristine check ──────────────────────────────────────────
_step "a: env capture + .so md5 check"
nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader > "$OUTDIR/nvidia_smi.txt" 2>&1
_so_md5 > "$OUTDIR/so_md5.txt"
grep -q "$SO_GOOD_MD5" "$OUTDIR/so_md5.txt"
_PRISTINE_OK=$?
echo "  pristine_md5_ok=$([ $_PRISTINE_OK -eq 0 ] && echo YES || echo NO): $(cat "$OUTDIR/so_md5.txt")"
if [ $_PRISTINE_OK -ne 0 ]; then
    echo "  WARNING: .so md5 != $SO_GOOD_MD5 — rebuild pristine before trusting capture verdicts"
fi

# ── b. integration gate cw through src dispatch (L2 default flip re-gate) ────
_step "b1: integration gate cw (native grid, default stage-B, L2-default K#1)"
eval "$PP" "$PY" scratch/gdn2_integration_box_cw.py \
    --out "$OUTDIR/integration_cw_l2.json" > "$OUTDIR/integration_cw_l2.log" 2>&1
_rc "b1" "$?"
_B1_GO=$(grep -o '"GO": true' "$OUTDIR/integration_cw_l2.json" | head -1)

_step "b2: integration gate cw --closed (Level-1b + L2 K#1)"
eval "$PP" "$PY" scratch/gdn2_integration_box_cw.py --closed \
    --out "$OUTDIR/integration_cw_closed_l2.json" > "$OUTDIR/integration_cw_closed_l2.log" 2>&1
_rc "b2" "$?"

L2FLAG=""
if [ -z "$_B1_GO" ]; then
    echo "  b1 NOT GO — falling back to FMR_DISABLE_L2=1 for remaining steps"
    L2FLAG="FMR_DISABLE_L2=1"
    _step "b3: integration gate cw RETRY with FMR_DISABLE_L2=1 (isolate the flip)"
    eval "$PP" FMR_DISABLE_L2=1 "$PY" scratch/gdn2_integration_box_cw.py \
        --out "$OUTDIR/integration_cw_nol2.json" > "$OUTDIR/integration_cw_nol2.log" 2>&1
    _rc "b3" "$?"
fi

# ── c. tiny-train native (the NaN-fix confirm) ────────────────────────────────
_step "c: tiny GDN-2 training, native arm (drifted-regime NaN fix)"
eval "$PP" $L2FLAG "$PY" scratch/gdn2_tiny_train.py --arm native \
    --out "$OUTDIR/tiny_train_native_v2.json" > "$OUTDIR/tiny_train_native_v2.log" 2>&1
_rc "c" "$?"
tail -3 "$OUTDIR/tiny_train_native_v2.log"

# ── d. graph re-bench v4 (event-path; closed arm; vs fla) ────────────────────
_step "d: graph bench v4 (4 shapes, event-path)"
eval "$PP" $L2FLAG "$PY" scratch/gdn2_graph_bench.py \
    --out "$OUTDIR/gdn2_graph_bench_v4.json" > "$OUTDIR/gdn2_graph_bench_v4.log" 2>&1
_rc "d" "$?"

# ── e. Level-3 v3 gates (unrolled fused kernel; arbiter staging nt=1 -> 4 -> 8) ──
# nt=1 = single acquire/commit cycle (equivalent to the proven straight-line body);
# nt=4 adjudicates multi-cycle pipeline phase behavior; nt=8 probes compile-time
# growth (record wall time from the logs; nt=32 only if it extrapolates sanely).
_step "e1: L3 v3 scalar nt=1"
timeout 1800 bash -c "$PP $L2FLAG time $PY scratch/k1_incb2_v3_unroll.py --mode scalar --nt 1 \
    --out $OUTDIR/k1_incb2_v3_scalar_nt1.json" > "$OUTDIR/k1_incb2_v3_scalar_nt1.log" 2>&1
_rc "e1" "$?"

_step "e2: L3 v3 scalar nt=4"
timeout 3600 bash -c "$PP $L2FLAG time $PY scratch/k1_incb2_v3_unroll.py --mode scalar --nt 4 \
    --out $OUTDIR/k1_incb2_v3_scalar_nt4.json" > "$OUTDIR/k1_incb2_v3_scalar_nt4.log" 2>&1
_rc "e2" "$?"

_step "e3: L3 v3 cw nt=4"
timeout 3600 bash -c "$PP $L2FLAG time $PY scratch/k1_incb2_v3_unroll.py --mode cw --nt 4 \
    --out $OUTDIR/k1_incb2_v3_cw_nt4.json" > "$OUTDIR/k1_incb2_v3_cw_nt4.log" 2>&1
_rc "e3" "$?"

_step "e4: L3 v3 cw nt=8 (compile-scale probe; only meaningful if e3 GO)"
timeout 3600 bash -c "$PP $L2FLAG time $PY scratch/k1_incb2_v3_unroll.py --mode cw --nt 8 \
    --out $OUTDIR/k1_incb2_v3_cw_nt8.json" > "$OUTDIR/k1_incb2_v3_cw_nt8.log" 2>&1
_rc "e4" "$?"

# ── f. promoted-L2 module re-gate (src import path) ──────────────────────────
_step "f: k1_l2_epilogue gate via promoted src module"
eval "$PP" "$PY" scratch/k1_l2_epilogue.py --mode cw \
    --out "$OUTDIR/k1_l2_epilogue_srcpath.json" > "$OUTDIR/k1_l2_epilogue_srcpath.log" 2>&1
_rc "f" "$?"

_step "SUMMARY"
for j in "$OUTDIR"/*.json; do
    echo "--- $j"
    "$PY" - "$j" <<'PYEOF' 2>/dev/null || true
import json, sys
d = json.load(open(sys.argv[1]))
keys = [k for k in ("GO", "gate_ok", "deterministic", "error") if k in d]
print("   ", {k: d[k] for k in keys})
PYEOF
done
echo ""
echo "burst-3 complete. Pull $OUTDIR + stop the box (--discard-local-ssd)."
