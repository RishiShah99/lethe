#!/usr/bin/env bash
# Box-side burst-7 orchestration — POST-REVIEW VERIFICATION (20-finding fix campaign).
# Runs from the repo root on the B200 box.  JSON in ~/box_out_burst7/.
#
# Usage:
#   nohup bash scratch/run_burst7.sh > ~/box_out_burst7/burst7.log 2>&1 &
#
#   a. env capture + pristine .so md5 check
#   b. k2f + sbe micro-gate re-gates (dispatch/assembly changed at desk; the
#      kernels themselves did not — this pins that on silicon)
#   c. integration gates: dv64 default + closed native grid + L2048 closed
#      (covers the create_graph dispatch guards and detach change)
#   d. tiny-train native, full default stack (drift regression + fallback==0
#      proves the do.requires_grad guard never misfires in real training)
#   e. FULL PYTEST SUITE on silicon — the 117 CUDA-skips go live, including
#      the rewritten C1/C2 bench smoke asserts (event-path timing) and the
#      memory-aware scan-mode selector paths.
#   No new kernels this burst: any FAIL is a fix-campaign regression — record,
#   pull, stop the box. Nothing here should need a second round.

set -o pipefail
OUTDIR=~/box_out_burst7
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
"$PY" -m pytest --version > /dev/null 2>&1 \
    || ~/.local/bin/uv pip install --python "$PY" pytest > "$OUTDIR/pytest_install.log" 2>&1

_step "b1: k2f micro-gate nt=32 b2h8 (re-gate under the fixed tree)"
timeout 3600 bash -c "$PP $PY scratch/k2f_microgate.py --nt 32 --bh 2,8 \
    --out $OUTDIR/k2f_microgate_nt32.json" > "$OUTDIR/k2f_microgate_nt32.log" 2>&1
_rc "b1" "$?"

_step "b2: sbe micro-gate nt=32 b2h8 (re-gate under the fixed tree)"
timeout 3600 bash -c "$PP $PY scratch/sbe_microgate.py --nt 32 --bh 2,8 \
    --out $OUTDIR/sbe_microgate_nt32.json" > "$OUTDIR/sbe_microgate_nt32.log" 2>&1
_rc "b2" "$?"

_step "c1: integration cw dv64 default native grid"
eval "$PP" "$PY" scratch/gdn2_integration_box_cw.py --dv 64 \
    --out "$OUTDIR/integration_cw_dv64_default.json" > "$OUTDIR/integration_cw_dv64_default.log" 2>&1
_rc "c1" "$?"

_step "c2: integration cw dv64 --closed native grid"
eval "$PP" "$PY" scratch/gdn2_integration_box_cw.py --dv 64 --closed \
    --out "$OUTDIR/integration_cw_dv64_closed.json" > "$OUTDIR/integration_cw_dv64_closed.log" 2>&1
_rc "c2" "$?"

_step "c3: integration cw dv64 --closed L2048"
eval "$PP" "$PY" scratch/gdn2_integration_box_cw.py --dv 64 --closed --shapes "2,2048,8" \
    --out "$OUTDIR/integration_cw_dv64_L2048_closed.json" > "$OUTDIR/integration_cw_dv64_L2048_closed.log" 2>&1
_rc "c3" "$?"

_step "d: tiny GDN-2 training, native arm, full default stack"
eval "$PP" "$PY" scratch/gdn2_tiny_train.py --arm native \
    --out "$OUTDIR/tiny_train_native_postfix.json" > "$OUTDIR/tiny_train_native_postfix.log" 2>&1
_rc "d" "$?"
tail -3 "$OUTDIR/tiny_train_native_postfix.log"

_step "e: FULL PYTEST SUITE on silicon"
timeout 10800 bash -c "$PP $PY -m pytest tests -q" > "$OUTDIR/pytest_gpu.log" 2>&1
_rc "e" "$?"
tail -15 "$OUTDIR/pytest_gpu.log"

_step "SUMMARY"
for j in "$OUTDIR"/*.json; do
    echo "--- $j"
    "$PY" - "$j" <<'PYEOF' 2>/dev/null || true
import json, sys
d = json.load(open(sys.argv[1]))
keys = [k for k in ("GO", "gate_ok", "deterministic", "worst_scale_rel", "error") if k in d]
print("   ", {k: d[k] for k in keys})
PYEOF
done
echo ""
echo "burst-7 complete. Pull $OUTDIR + stop the box (--discard-local-ssd)."
