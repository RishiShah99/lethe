#!/usr/bin/env bash
# Box-side burst-6 orchestration — THE SQUEEZE (stage-B einsum kernel) + final re-bench.
# Runs from the repo root on the B200 box.  JSON in ~/box_out_burst6/.
#
# Usage:
#   nohup bash scratch/run_burst6.sh > ~/box_out_burst6/burst6.log 2>&1 &
#
#   a. env capture + pristine .so md5 check
#   b. sbe micro-gates: nt=4 b2h2 -> nt=32 b2h8 (Z=512) + drift leg (gscale=40)
#   c. sbe rung bench: kernel vs torch (decay_rel build + 2 einsums)
#   KILL: if b not GO, stop (FMR_DISABLE_SBE=1 keeps the torch path — burst-5
#   numbers stand; the squeeze is then a recorded negative).
#   d. integration gates with SBE live: dv64 closed native grid + L2048 closed
#      (SBE only fires on the closed no-graph path; dv128 control is unaffected
#      by dims — skip, burst-5 e4 stands).
#   e. FINAL RE-BENCH: graph bench dv64 with K2F+SBE default (+ savefwd arm).
#      Burst-5 f1 (SBE absent) is the control.
#   f. tiny-train native under the full default stack (drift regression).

set -o pipefail
OUTDIR=~/box_out_burst6
mkdir -p "$OUTDIR"
PY=~/cuteenv/bin/python
PP="PYTHONPATH=src:."
SO_GOOD_MD5=a1e3ba0a1e62227bccbea2aaf20cb6e8

_step() { echo ""; echo "================================================================"; echo "  STEP: $1"; echo "================================================================"; }
_rc()   { echo "  [$1] exit_code=$2"; }
_go()   { grep -qo '"GO": true' "$1" 2>/dev/null; }

_step "a: env capture + .so md5 check"
nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader > "$OUTDIR/nvidia_smi.txt" 2>&1
find ~/cuteenv/lib/python3.12/site-packages/nvidia_cutlass_dsl -name 'libcute_dsl_runtime.so' \
    -exec md5sum {} \; 2>/dev/null | head -1 > "$OUTDIR/so_md5.txt"
grep -q "$SO_GOOD_MD5" "$OUTDIR/so_md5.txt" \
    && echo "  pristine_md5_ok=YES" || echo "  WARNING: .so md5 mismatch — rebuild pristine"

_step "b1: sbe micro-gate nt=4 b2h2"
timeout 3600 bash -c "$PP $PY scratch/sbe_microgate.py --nt 4 --bh 2,2 \
    --out $OUTDIR/sbe_microgate_nt4.json" > "$OUTDIR/sbe_microgate_nt4.log" 2>&1
_rc "b1" "$?"

_step "b2: sbe micro-gate nt=32 b2h8 (Z=512)"
timeout 3600 bash -c "$PP $PY scratch/sbe_microgate.py --nt 32 --bh 2,8 \
    --out $OUTDIR/sbe_microgate_nt32_b2h8.json" > "$OUTDIR/sbe_microgate_nt32_b2h8.log" 2>&1
_rc "b2" "$?"

_step "b3: sbe micro-gate DRIFT nt=32 b2h8"
timeout 3600 bash -c "$PP $PY scratch/sbe_microgate.py --drift --nt 32 --bh 2,8 \
    --out $OUTDIR/sbe_microgate_drift_nt32.json" > "$OUTDIR/sbe_microgate_drift_nt32.log" 2>&1
_rc "b3" "$?"

_step "c: sbe rung bench nt=32 b2h8"
eval "$PP" "$PY" scratch/sbe_microgate.py --bench --nt 32 --bh 2,8 \
    --out "$OUTDIR/sbe_rung_bench_nt32_b2h8.json" > "$OUTDIR/sbe_rung_bench_nt32_b2h8.log" 2>&1
_rc "c" "$?"

if ! _go "$OUTDIR/sbe_microgate_nt4.json" || ! _go "$OUTDIR/sbe_microgate_nt32_b2h8.json"; then
    echo ""
    echo "KILL: sbe micro-gate NOT GO — skipping integration/re-bench (FMR_DISABLE_SBE=1"
    echo "keeps the torch path; burst-5 numbers stand; the squeeze is a recorded negative)."
    echo "burst-6 stopped at the kill check. Pull $OUTDIR + stop the box (--discard-local-ssd)."
    exit 0
fi

_step "d1: integration cw dv64 --closed native grid (SBE live)"
eval "$PP" "$PY" scratch/gdn2_integration_box_cw.py --dv 64 --closed \
    --out "$OUTDIR/integration_cw_dv64_closed_sbe.json" > "$OUTDIR/integration_cw_dv64_closed_sbe.log" 2>&1
_rc "d1" "$?"

SBEFLAG=""
if ! _go "$OUTDIR/integration_cw_dv64_closed_sbe.json"; then
    echo "  d1 NOT GO — isolating with FMR_DISABLE_SBE=1"
    SBEFLAG="FMR_DISABLE_SBE=1"
    eval "$PP" FMR_DISABLE_SBE=1 "$PY" scratch/gdn2_integration_box_cw.py --dv 64 --closed \
        --out "$OUTDIR/integration_cw_dv64_closed_nosbe.json" > "$OUTDIR/integration_cw_dv64_closed_nosbe.log" 2>&1
    _rc "d1b" "$?"
fi

_step "d2: integration cw dv64 --closed L2048 (SBE live)"
eval "$PP" $SBEFLAG "$PY" scratch/gdn2_integration_box_cw.py --dv 64 --closed --shapes "2,2048,8" \
    --out "$OUTDIR/integration_cw_dv64_L2048_closed_sbe.json" > "$OUTDIR/integration_cw_dv64_L2048_closed_sbe.log" 2>&1
_rc "d2" "$?"

_step "e: FINAL RE-BENCH graph bench dv64 (K2F + SBE default; savefwd arm rides)"
timeout 5400 bash -c "$PP $SBEFLAG $PY scratch/gdn2_graph_bench.py --dv 64 \
    --out $OUTDIR/gdn2_graph_bench_dv64_k2f_sbe.json" > "$OUTDIR/gdn2_graph_bench_dv64_k2f_sbe.log" 2>&1
_rc "e" "$?"

_step "f: tiny GDN-2 training, native arm, full default stack"
eval "$PP" $SBEFLAG "$PY" scratch/gdn2_tiny_train.py --arm native \
    --out "$OUTDIR/tiny_train_native_sbe.json" > "$OUTDIR/tiny_train_native_sbe.log" 2>&1
_rc "f" "$?"
tail -3 "$OUTDIR/tiny_train_native_sbe.log"

_step "SUMMARY"
for j in "$OUTDIR"/*.json; do
    echo "--- $j"
    "$PY" - "$j" <<'PYEOF' 2>/dev/null || true
import json, sys
d = json.load(open(sys.argv[1]))
keys = [k for k in ("GO", "gate_ok", "deterministic", "error") if k in d]
print("   ", {k: d[k] for k in keys})
if "runs" in d:
    for r in d["runs"]:
        picks = {k: r.get(k) for k in ("shape", "closed_graph_ms", "savefwd_graph_ms",
                 "fla_ms", "closed_graph_over_fla", "savefwd_graph_over_fla") if r.get(k) is not None}
        print("   ", picks)
PYEOF
done
echo ""
echo "burst-6 complete. Pull $OUTDIR + stop the box (--discard-local-ssd)."
