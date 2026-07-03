#!/usr/bin/env bash
# Box-side burst-5 orchestration — THE K#2-FUSION CAMPAIGN burst.  Runs from the
# repo root on the B200 box.  Each step independent (|| true); JSON in ~/box_out_burst5/.
#
# Usage:
#   nohup bash scratch/run_burst5.sh > ~/box_out_burst5/burst5.log 2>&1 &
#
# Dependency order (HANDOFF campaign spec):
#   a. env capture + pristine .so md5 check (a1e3ba0a — the burst-2 control)
#   b. K#2-fused micro-gates (scratch/k2f_microgate.py): nt=4 b2h2 -> nt=8 b2h2 ->
#      nt=32 b2h8 (bench scale; grid-z design = no compile wall, one executable per
#      Z).  GO = all 5 grads scale_rel < 5e-3 + 2-run bit-determinism.
#   c. drifted-regime micro-gates (--drift, gscale=40: within-chunk log2 span > 128)
#      at nt=4 and nt=32 — extends the c707201 regression to the fused path's fp16
#      mid-chain landings of dT/X.
#   d. K#2 rung bench (--bench): fused vs lever-B batched (vs per-chunk serial at
#      the small shape) at nt=8 b2h2 and nt=32 b2h8.  Event-path only.
#   KILL: if b is not GO after c-leg diagnostics, stop here (campaign kill
#   criterion: micro-gate not GO by box round 2 -> stop).  The remaining legs run
#   only on GO.
#   e. integration gates through src dispatch with the K2F-DEFAULT selector:
#      dv=64 native grid default+closed, L2048 default+closed; dv=128 control
#      (K2F dim-skipped).  On e1 failure: isolate with FMR_DISABLE_K2F=1.
#   f. THE DECISION GATE: graph bench dv=64 K2F-on vs FMR_DISABLE_K2F=1 control.
#      Captured closed+graph @2x2048x8 dv64: <=42 ms = decomposition VALIDATED;
#      >=48 ms with a correct kernel = FALSIFIED (record honestly); between =
#      partial.  The bench also carries the NEW savefwd arm (lever 2b: stash built
#      outside the timed region + stash_build_ms) and lever 2a rides the closed arm
#      (decay_rel built once).
#   g. tiny-train native under the K2F default — the strongest drift regression
#      (300 steps; backward_dispatch counters prove native carried every step).

set -o pipefail
OUTDIR=~/box_out_burst5
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

_go() { grep -qo '"GO": true' "$1" 2>/dev/null; }

# ── a. env capture + pristine check ──────────────────────────────────────────
_step "a: env capture + .so md5 check"
nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader > "$OUTDIR/nvidia_smi.txt" 2>&1
_so_md5 > "$OUTDIR/so_md5.txt"
grep -q "$SO_GOOD_MD5" "$OUTDIR/so_md5.txt"
_PRISTINE_OK=$?
echo "  pristine_md5_ok=$([ $_PRISTINE_OK -eq 0 ] && echo YES || echo NO): $(cat "$OUTDIR/so_md5.txt")"
if [ $_PRISTINE_OK -ne 0 ]; then
    echo "  WARNING: .so md5 != $SO_GOOD_MD5 — rebuild pristine before trusting verdicts"
fi

# ── b. K#2-fused micro-gates ──────────────────────────────────────────────────
_S0=$SECONDS
_step "b1: k2f micro-gate nt=4 b2h2"
timeout 3600 bash -c "$PP $PY scratch/k2f_microgate.py --nt 4 --bh 2,2 \
    --out $OUTDIR/k2f_microgate_nt4.json" > "$OUTDIR/k2f_microgate_nt4.log" 2>&1
_rc "b1" "$?"
echo "  elapsed_s=$((SECONDS - _S0))"

# Skeptic finding 1 fallback: if b1 fails (likely the triangular tail's data-dependent
# bounds — the one no-precedent DSL shape), retry with the wheel-proven predicated tail.
K2FTAIL=""
if ! _go "$OUTDIR/k2f_microgate_nt4.json"; then
    _step "b1b: k2f micro-gate nt=4 RETRY with FMR_K2F_PRED_TAIL=1 (SegSum-idiom tail)"
    timeout 3600 bash -c "$PP FMR_K2F_PRED_TAIL=1 $PY scratch/k2f_microgate.py --nt 4 --bh 2,2 \
        --out $OUTDIR/k2f_microgate_nt4_pred.json" > "$OUTDIR/k2f_microgate_nt4_pred.log" 2>&1
    _rc "b1b" "$?"
    if _go "$OUTDIR/k2f_microgate_nt4_pred.json"; then
        echo "  predicated tail GO — all remaining legs run with FMR_K2F_PRED_TAIL=1"
        K2FTAIL="FMR_K2F_PRED_TAIL=1"
        cp "$OUTDIR/k2f_microgate_nt4_pred.json" "$OUTDIR/k2f_microgate_nt4.json"
    fi
fi

_step "b2: k2f micro-gate nt=8 b2h2"
timeout 3600 bash -c "$PP $K2FTAIL $PY scratch/k2f_microgate.py --nt 8 --bh 2,2 \
    --out $OUTDIR/k2f_microgate_nt8.json" > "$OUTDIR/k2f_microgate_nt8.log" 2>&1
_rc "b2" "$?"

_step "b3: k2f micro-gate nt=32 b2h8 (bench scale, Z=512)"
timeout 3600 bash -c "$PP $K2FTAIL $PY scratch/k2f_microgate.py --nt 32 --bh 2,8 \
    --out $OUTDIR/k2f_microgate_nt32_b2h8.json" > "$OUTDIR/k2f_microgate_nt32_b2h8.log" 2>&1
_rc "b3" "$?"

# ── c. drifted-regime micro-gates ─────────────────────────────────────────────
_step "c1: k2f micro-gate DRIFT nt=4"
timeout 3600 bash -c "$PP $K2FTAIL $PY scratch/k2f_microgate.py --drift --nt 4 --bh 2,2 \
    --out $OUTDIR/k2f_microgate_drift_nt4.json" > "$OUTDIR/k2f_microgate_drift_nt4.log" 2>&1
_rc "c1" "$?"

_step "c2: k2f micro-gate DRIFT nt=32 b2h8"
timeout 3600 bash -c "$PP $K2FTAIL $PY scratch/k2f_microgate.py --drift --nt 32 --bh 2,8 \
    --out $OUTDIR/k2f_microgate_drift_nt32.json" > "$OUTDIR/k2f_microgate_drift_nt32.log" 2>&1
_rc "c2" "$?"

# ── d. K#2 rung bench ─────────────────────────────────────────────────────────
_step "d1: k2f rung bench nt=8 b2h2 (incl. per-chunk serial)"
eval "$PP" $K2FTAIL "$PY" scratch/k2f_microgate.py --bench --nt 8 --bh 2,2 \
    --out "$OUTDIR/k2f_rung_bench_nt8_b2h2.json" > "$OUTDIR/k2f_rung_bench_nt8_b2h2.log" 2>&1
_rc "d1" "$?"

_step "d2: k2f rung bench nt=32 b2h8 (the L2048 bench group count)"
eval "$PP" $K2FTAIL "$PY" scratch/k2f_microgate.py --bench --nt 32 --bh 2,8 \
    --out "$OUTDIR/k2f_rung_bench_nt32_b2h8.json" > "$OUTDIR/k2f_rung_bench_nt32_b2h8.log" 2>&1
_rc "d2" "$?"

# ── KILL CHECK: remaining legs only if the micro-gates are GO ─────────────────
if ! _go "$OUTDIR/k2f_microgate_nt4.json" || ! _go "$OUTDIR/k2f_microgate_nt32_b2h8.json"; then
    echo ""
    echo "KILL: k2f micro-gate NOT GO — skipping integration/graph legs (campaign kill criterion)."
    echo "Diagnose from the c-leg + logs; fallback = FMR_DISABLE_K2F=1 (batched default unchanged)."
    echo "burst-5 stopped at the kill check. Pull $OUTDIR + stop the box (--discard-local-ssd)."
    exit 0
fi

# ── e. integration gates (K2F-default selector through src dispatch) ──────────
_step "e1: integration cw dv=64 native grid (default stage-B; K2F live)"
eval "$PP" $K2FTAIL "$PY" scratch/gdn2_integration_box_cw.py --dv 64 \
    --out "$OUTDIR/integration_cw_dv64_k2f.json" > "$OUTDIR/integration_cw_dv64_k2f.log" 2>&1
_rc "e1" "$?"

K2FFLAG=""
if ! _go "$OUTDIR/integration_cw_dv64_k2f.json"; then
    echo "  e1 NOT GO — isolating with FMR_DISABLE_K2F=1, remaining dv=64 legs fall back"
    K2FFLAG="FMR_DISABLE_K2F=1"
    _step "e1b: integration cw dv=64 RETRY with FMR_DISABLE_K2F=1 (isolate the flip)"
    eval "$PP" FMR_DISABLE_K2F=1 "$PY" scratch/gdn2_integration_box_cw.py --dv 64 \
        --out "$OUTDIR/integration_cw_dv64_nok2f.json" > "$OUTDIR/integration_cw_dv64_nok2f.log" 2>&1
    _rc "e1b" "$?"
fi

_step "e2: integration cw dv=64 --closed (K2F + closed stage-B + lever 2a)"
eval "$PP" $K2FFLAG $K2FTAIL "$PY" scratch/gdn2_integration_box_cw.py --dv 64 --closed \
    --out "$OUTDIR/integration_cw_dv64_closed_k2f.json" > "$OUTDIR/integration_cw_dv64_closed_k2f.log" 2>&1
_rc "e2" "$?"

_step "e3: integration cw dv=64 L2048 default + closed"
eval "$PP" $K2FFLAG $K2FTAIL "$PY" scratch/gdn2_integration_box_cw.py --dv 64 --shapes "2,2048,8" \
    --out "$OUTDIR/integration_cw_dv64_L2048_k2f.json" > "$OUTDIR/integration_cw_dv64_L2048_k2f.log" 2>&1
_rc "e3" "$?"
eval "$PP" $K2FFLAG $K2FTAIL "$PY" scratch/gdn2_integration_box_cw.py --dv 64 --closed --shapes "2,2048,8" \
    --out "$OUTDIR/integration_cw_dv64_L2048_closed_k2f.json" > "$OUTDIR/integration_cw_dv64_L2048_closed_k2f.log" 2>&1
_rc "e3-closed" "$?"

_step "e4: integration cw dv=128 default + closed (K2F dim-skipped — no-regression control)"
eval "$PP" "$PY" scratch/gdn2_integration_box_cw.py \
    --out "$OUTDIR/integration_cw_dv128_k2fwire.json" > "$OUTDIR/integration_cw_dv128_k2fwire.log" 2>&1
_rc "e4" "$?"
eval "$PP" "$PY" scratch/gdn2_integration_box_cw.py --closed \
    --out "$OUTDIR/integration_cw_dv128_closed_k2fwire.json" > "$OUTDIR/integration_cw_dv128_closed_k2fwire.log" 2>&1
_rc "e4-closed" "$?"

# ── f. THE DECISION GATE: graph bench dv=64 (K2F on vs off; savefwd arm rides) ─
_step "f1: graph bench dv=64 with K2F default (+ savefwd arm)"
timeout 5400 bash -c "$PP $K2FFLAG $K2FTAIL $PY scratch/gdn2_graph_bench.py --dv 64 \
    --out $OUTDIR/gdn2_graph_bench_dv64_k2f.json" > "$OUTDIR/gdn2_graph_bench_dv64_k2f.log" 2>&1
_rc "f1" "$?"

_step "f2: graph bench dv=64 with FMR_DISABLE_K2F=1 (the batched-K#2 control)"
timeout 5400 bash -c "$PP FMR_DISABLE_K2F=1 $PY scratch/gdn2_graph_bench.py --dv 64 \
    --out $OUTDIR/gdn2_graph_bench_dv64_nok2f.json" > "$OUTDIR/gdn2_graph_bench_dv64_nok2f.log" 2>&1
_rc "f2" "$?"

# ── g. tiny-train native under the K2F default (drift regression, strongest) ──
_step "g: tiny GDN-2 training, native arm, K2F default"
eval "$PP" $K2FFLAG $K2FTAIL "$PY" scratch/gdn2_tiny_train.py --arm native \
    --out "$OUTDIR/tiny_train_native_k2f.json" > "$OUTDIR/tiny_train_native_k2f.log" 2>&1
_rc "g" "$?"
tail -3 "$OUTDIR/tiny_train_native_k2f.log"

_step "SUMMARY"
for j in "$OUTDIR"/*.json; do
    echo "--- $j"
    "$PY" - "$j" <<'PYEOF' 2>/dev/null || true
import json, sys
d = json.load(open(sys.argv[1]))
keys = [k for k in ("GO", "gate_ok", "deterministic", "backward_dispatch", "error") if k in d]
print("   ", {k: d[k] for k in keys})
if "runs" in d:
    for r in d["runs"]:
        picks = {k: r.get(k) for k in ("shape", "closed_graph_ms", "savefwd_graph_ms",
                 "stash_build_ms", "fla_ms", "closed_graph_over_fla") if r.get(k) is not None}
        print("   ", picks)
PYEOF
done
echo ""
echo "DECISION GATE: closed_graph_ms @ [2,2048,8] dv64 — <=42 VALIDATED / >=48 FALSIFIED / between PARTIAL"
echo "burst-5 complete. Pull $OUTDIR + stop the box (--discard-local-ssd)."
