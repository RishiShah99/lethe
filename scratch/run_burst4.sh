#!/usr/bin/env bash
# Box-side burst-4 orchestration.  Runs from the repo root on the B200 box.
# Each step independent (|| true); JSON in ~/box_out_burst4/.
#
# Usage:
#   nohup bash scratch/run_burst4.sh > ~/box_out_burst4/burst4.log 2>&1 &
#
# Dependency order (HANDOFF BURST 4 LIST):
#   a. env capture + pristine .so md5 check (a1e3ba0a — the burst-2 control)
#   b. L3 v3 compile-scale probes cw nt=16 then nt=32 (L2048 = NT 32 @ chunk 64);
#      the harness now imports the kernel from src -> this is also the src-path
#      re-gate. If nt=32 is not GO, every nt=32-shaped leg below falls back to
#      FMR_DISABLE_L3=1 (the selector kill-switch).
#   c. integration gates through src dispatch with the L3-DEFAULT selector:
#      dv=64 (L3 live) native grid default+closed, then L2048; dv=128 (L3
#      dim-skipped) default+closed — the no-regression control at the crown dims.
#      If c1 FAILS, later dv=64 steps re-run with FMR_DISABLE_L3=1.
#   d. THE RUNG BENCH: K#1-only event-path ladder (L3 fused vs L2 vs lever-B)
#      at nt=8 / nt=32 (b2h2) and nt=32 (b2h8 = the L2048 bench group count).
#   e. THE SPEED QUESTION: graph bench at dv=64, L3-on vs FMR_DISABLE_L3=1
#      (same shapes incl. the fla bar). L3 arm capped at nt<=32 shapes (nt=64
#      unroll is unprobed); the L2 baseline runs all 4 shapes.
#   f. tiny-train native with the L3 default — re-purchases "trains" for the
#      fused kernel in the drifted regime (burst-3's confirm ran on L2).
#   g. FAMILY training arms gla/ssd/kda (300 steps) + la (600 steps — weakest
#      mixer for delayed-copy, converges slower; gate criterion unchanged),
#      pinned to FMR_DISABLE_L3=1 (the burst-3-proven L2 config) so the family
#      purchase is unconfounded by the new default. backward_dispatch in each
#      JSON proves the native path carried every backward.
#
# Deferred this burst (carried): mamba_ssm install + scan.cu re-measure (CUDA
# extension build, heavy wall; archive item) · loop_tile SIGSEGV (low priority).

set -o pipefail
OUTDIR=~/box_out_burst4
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

# ── b. L3 compile-scale probes (cw nt=16 -> 32; also the src-path re-gate) ───
_S0=$SECONDS
_step "b1: L3 v3 cw nt=16 (src-path)"
timeout 3600 bash -c "$PP $PY scratch/k1_incb2_v3_unroll.py --mode cw --nt 16 \
    --out $OUTDIR/k1_incb2_v3_cw_nt16.json" > "$OUTDIR/k1_incb2_v3_cw_nt16.log" 2>&1
_rc "b1" "$?"
echo "  elapsed_s=$((SECONDS - _S0))"

_S0=$SECONDS
_step "b2: L3 v3 cw nt=32 (bench scale; compile-wall stress)"
timeout 3600 bash -c "$PP $PY scratch/k1_incb2_v3_unroll.py --mode cw --nt 32 \
    --out $OUTDIR/k1_incb2_v3_cw_nt32.json" > "$OUTDIR/k1_incb2_v3_cw_nt32.log" 2>&1
_rc "b2" "$?"
echo "  elapsed_s=$((SECONDS - _S0))"

NT32FLAG=""
if ! _go "$OUTDIR/k1_incb2_v3_cw_nt32.json"; then
    echo "  nt=32 NOT GO — nt=32-shaped legs below run with FMR_DISABLE_L3=1"
    NT32FLAG="FMR_DISABLE_L3=1"
fi

# ── c. integration gates (L3-default selector through src dispatch) ──────────
_step "c1: integration cw dv=64 native grid (default stage-B; L3 live)"
eval "$PP" "$PY" scratch/gdn2_integration_box_cw.py --dv 64 \
    --out "$OUTDIR/integration_cw_dv64_l3.json" > "$OUTDIR/integration_cw_dv64_l3.log" 2>&1
_rc "c1" "$?"

L3FLAG=""
if ! _go "$OUTDIR/integration_cw_dv64_l3.json"; then
    echo "  c1 NOT GO — isolating with FMR_DISABLE_L3=1, remaining dv=64 legs fall back"
    L3FLAG="FMR_DISABLE_L3=1"
    _step "c1b: integration cw dv=64 RETRY with FMR_DISABLE_L3=1 (isolate the flip)"
    eval "$PP" FMR_DISABLE_L3=1 "$PY" scratch/gdn2_integration_box_cw.py --dv 64 \
        --out "$OUTDIR/integration_cw_dv64_nol3.json" > "$OUTDIR/integration_cw_dv64_nol3.log" 2>&1
    _rc "c1b" "$?"
fi

_step "c2: integration cw dv=64 --closed (Level-1b + L3 K#1)"
eval "$PP" $L3FLAG "$PY" scratch/gdn2_integration_box_cw.py --dv 64 --closed \
    --out "$OUTDIR/integration_cw_dv64_closed_l3.json" > "$OUTDIR/integration_cw_dv64_closed_l3.log" 2>&1
_rc "c2" "$?"

_step "c3: integration cw dv=64 L2048 (nt=32 shape)"
eval "$PP" $NT32FLAG $L3FLAG "$PY" scratch/gdn2_integration_box_cw.py --dv 64 \
    --shapes "2,2048,8" \
    --out "$OUTDIR/integration_cw_dv64_L2048.json" > "$OUTDIR/integration_cw_dv64_L2048.log" 2>&1
_rc "c3" "$?"

_step "c4: integration cw dv=64 L2048 --closed"
eval "$PP" $NT32FLAG $L3FLAG "$PY" scratch/gdn2_integration_box_cw.py --dv 64 --closed \
    --shapes "2,2048,8" \
    --out "$OUTDIR/integration_cw_dv64_L2048_closed.json" > "$OUTDIR/integration_cw_dv64_L2048_closed.log" 2>&1
_rc "c4" "$?"

_step "c5: integration cw dv=128 default (L3 dim-skipped — no-regression control)"
eval "$PP" "$PY" scratch/gdn2_integration_box_cw.py \
    --out "$OUTDIR/integration_cw_dv128.json" > "$OUTDIR/integration_cw_dv128.log" 2>&1
_rc "c5" "$?"

_step "c6: integration cw dv=128 --closed"
eval "$PP" "$PY" scratch/gdn2_integration_box_cw.py --closed \
    --out "$OUTDIR/integration_cw_dv128_closed.json" > "$OUTDIR/integration_cw_dv128_closed.log" 2>&1
_rc "c6" "$?"

# ── d. the rung bench: K#1-only ladder timing ────────────────────────────────
_step "d1: rung bench nt=8 b2h2"
eval "$PP" "$PY" scratch/k1_incb2_v3_unroll.py --bench --nt 8 --bh 2,2 \
    --out "$OUTDIR/k1_rung_bench_nt8_b2h2.json" > "$OUTDIR/k1_rung_bench_nt8_b2h2.log" 2>&1
_rc "d1" "$?"

if [ -z "$NT32FLAG" ]; then
    _step "d2: rung bench nt=32 b2h2"
    eval "$PP" "$PY" scratch/k1_incb2_v3_unroll.py --bench --nt 32 --bh 2,2 \
        --out "$OUTDIR/k1_rung_bench_nt32_b2h2.json" > "$OUTDIR/k1_rung_bench_nt32_b2h2.log" 2>&1
    _rc "d2" "$?"

    _step "d3: rung bench nt=32 b2h8 (the L2048 bench group count)"
    eval "$PP" "$PY" scratch/k1_incb2_v3_unroll.py --bench --nt 32 --bh 2,8 \
        --out "$OUTDIR/k1_rung_bench_nt32_b2h8.json" > "$OUTDIR/k1_rung_bench_nt32_b2h8.log" 2>&1
    _rc "d3" "$?"
else
    echo "  d2/d3 skipped: nt=32 probe not GO"
fi

# ── e. THE SPEED QUESTION: graph bench dv=64, L3-on vs L3-off ────────────────
# L3-on capped at nt<=32 shapes (L4096 = nt 64, unprobed unroll); baseline runs all 4.
_step "e1: graph bench dv=64 with L3 default (nt<=32 shapes)"
timeout 5400 bash -c "$PP $NT32FLAG $L3FLAG $PY scratch/gdn2_graph_bench.py --dv 64 \
    --shapes 1x512x4,1x1024x8,2x2048x8 \
    --out $OUTDIR/gdn2_graph_bench_dv64_l3.json" > "$OUTDIR/gdn2_graph_bench_dv64_l3.log" 2>&1
_rc "e1" "$?"

_step "e2: graph bench dv=64 with FMR_DISABLE_L3=1 (the L2 baseline, all 4 shapes)"
timeout 5400 bash -c "$PP FMR_DISABLE_L3=1 $PY scratch/gdn2_graph_bench.py --dv 64 \
    --out $OUTDIR/gdn2_graph_bench_dv64_l2.json" > "$OUTDIR/gdn2_graph_bench_dv64_l2.log" 2>&1
_rc "e2" "$?"

# ── f. tiny-train native under the L3 default (drifted-regime confirm) ───────
_step "f: tiny GDN-2 training, native arm, L3 default"
eval "$PP" $L3FLAG "$PY" scratch/gdn2_tiny_train.py --arm native \
    --out "$OUTDIR/tiny_train_native_l3.json" > "$OUTDIR/tiny_train_native_l3.log" 2>&1
_rc "f" "$?"
tail -3 "$OUTDIR/tiny_train_native_l3.log"

# ── g. FAMILY training arms (pinned to the proven L2 config) ─────────────────
for FAM in gla ssd kda; do
    _step "g: tiny family training --arm $FAM (300 steps, FMR_DISABLE_L3=1)"
    eval "$PP" FMR_DISABLE_L3=1 "$PY" scratch/gdn2_tiny_train.py --arm "$FAM" --steps 300 \
        --out "$OUTDIR/tiny_train_${FAM}.json" > "$OUTDIR/tiny_train_${FAM}.log" 2>&1
    _rc "g:$FAM" "$?"
    tail -2 "$OUTDIR/tiny_train_${FAM}.log"
done

_step "g: tiny family training --arm la (600 steps, FMR_DISABLE_L3=1)"
eval "$PP" FMR_DISABLE_L3=1 "$PY" scratch/gdn2_tiny_train.py --arm la --steps 600 \
    --out "$OUTDIR/tiny_train_la.json" > "$OUTDIR/tiny_train_la.log" 2>&1
_rc "g:la" "$?"
tail -2 "$OUTDIR/tiny_train_la.log"

_step "SUMMARY"
for j in "$OUTDIR"/*.json; do
    echo "--- $j"
    "$PY" - "$j" <<'PYEOF' 2>/dev/null || true
import json, sys
d = json.load(open(sys.argv[1]))
keys = [k for k in ("GO", "gate_ok", "deterministic", "backward_dispatch", "error") if k in d]
print("   ", {k: d[k] for k in keys})
PYEOF
done
echo ""
echo "burst-4 complete. Pull $OUTDIR + stop the box (--discard-local-ssd)."
