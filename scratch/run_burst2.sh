#!/usr/bin/env bash
# Box-side burst-2 orchestration.  Runs from the repo root on the B200 box.
# Each step is independent (|| true); status captured and echoed; JSON in ~/box_out_burst2/.
#
# Usage:
#   nohup bash scratch/run_burst2.sh > ~/box_out_burst2/burst2.log 2>&1 &
#
# Dependency order (graph-capture verdict 2026-07-02: repo window EXONERATED at desk;
# the burst-1 failure matrix was confounded by the #3259 env perturbation):
#   b. PRISTINE cuteenv rebuild + .so md5 == a1e3ba0a...  (the discriminating control)
#   c. one-shape graph confirm (+ graph_loop_probe discriminator)
#   e. module A/B with audit_out/graphreg/good_*.py ONLY if c still fails
#   f. full graph re-bench (4 shapes; closed arm now runs the Level-1b batched restage)
#   g. Level-1b closed-path oracle gate (native grid + L2048)
#   h. Level-2 gate (epilogue-glue kernels)   i. Level-3 gates (fused offset-lifecycle)
#   j. tiny GDN-2 training run (purchases "trains the family")   k. fla Rosetta
#   l. family external parity rerun   m. mamba_ssm install + scan.cu re-measure

set -o pipefail
OUTDIR=~/box_out_burst2
mkdir -p "$OUTDIR"
PY=~/cuteenv/bin/python
PP=PYTHONPATH=src:.
UV=~/.local/bin/uv
SO_GOOD_MD5=a1e3ba0a1e62227bccbea2aaf20cb6e8

_step() { echo ""; echo "================================================================"; echo "  STEP: $1"; echo "================================================================"; }
_rc()   { echo "  [$1] exit_code=$2"; }

_so_md5() {
    find ~/cuteenv/lib/python3.12/site-packages/nvidia_cutlass_dsl -name 'libcute_dsl_runtime.so' \
        -exec md5sum {} \; 2>/dev/null | head -1
}

# ── a. env capture (state as found, BEFORE any change) ───────────────────────
_step "a: env capture (as found)"
nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader > "$OUTDIR/nvidia_smi.txt" 2>&1
_so_md5 > "$OUTDIR/so_md5_found.txt"
~/cuteenv/bin/pip list 2>/dev/null | grep -iE 'cutlass|cuda|triton|nvidia|torch|fla' > "$OUTDIR/pip_found.txt"
_rc "a" "$?"

# ── b. PRISTINE cuteenv rebuild ───────────────────────────────────────────────
_step "b: pristine rebuild (drop cu13 dist, pin cuda-pathfinder 1.5.5, reinstall libs-base)"
"$UV" pip uninstall --python "$PY" nvidia-cutlass-dsl-libs-cu13 > "$OUTDIR/pristine_rebuild.log" 2>&1 || true
"$UV" pip install --python "$PY" --force-reinstall cuda-pathfinder==1.5.5 nvidia-cutlass-dsl-libs-base==4.5.2 >> "$OUTDIR/pristine_rebuild.log" 2>&1
_RC_B=$?
_so_md5 > "$OUTDIR/so_md5_pristine.txt"
grep -q "$SO_GOOD_MD5" "$OUTDIR/so_md5_pristine.txt"
_PRISTINE_OK=$?
_rc "b(reinstall)" "$_RC_B"
echo "  pristine_md5_ok=$([ $_PRISTINE_OK -eq 0 ] && echo YES || echo NO): $(cat "$OUTDIR/so_md5_pristine.txt")"
if [ $_PRISTINE_OK -ne 0 ]; then
    echo "  WARNING: .so md5 != $SO_GOOD_MD5 — capture verdicts below are NOT pristine-controlled"
fi

# ── c. one-shape graph confirm (THE discriminating experiment) ────────────────
_step "c: graph capture one-shape confirm (pristine env)"
eval "$PP" CUDA_LAUNCH_BLOCKING=1 "$PY" scratch/gdn2_graph_bench.py \
    --shapes 1x512x4 --out "$OUTDIR/graphreg_pristine_confirm.json" \
    > "$OUTDIR/graphreg_pristine_confirm.log" 2>&1
_rc "c(confirm)" "$?"
grep -q '"graph_err"' "$OUTDIR/graphreg_pristine_confirm.json" 2>/dev/null
_CAPTURE_STILL_FAILS=$([ $? -eq 0 ] && echo 1 || echo 0)
echo "  capture_still_fails=$_CAPTURE_STILL_FAILS"

# ── d. minimal DSL-launch-in-capture discriminator ────────────────────────────
_step "d: graph_loop_probe (minimal capture discriminator)"
eval "$PP" "$PY" scratch/graph_loop_probe.py > "$OUTDIR/graph_loop_probe.log" 2>&1
_rc "d" "$?"

# ── e. module A/B (ONLY if c failed): 78715a4 snapshots over HEAD, then restore ─
# Swap ONLY the 4 src cute modules (harness constant → isolates the kernel modules;
# desk diff showed the harness changed by import-rewire + instrumentation only).
if [ "$_CAPTURE_STILL_FAILS" = "1" ] && [ -d audit_out/graphreg ]; then
    _step "e: module A/B (78715a4 good src snapshots, current harness)"
    CUTE=src/flash_mamba_rl/kernels/cute
    for f in gdn2_bwd_dhu gdn2_bwd_dhu_cw gdn2_bwd_wy gdn2_bwd_wy_cw; do
        cp "$CUTE/$f.py" "$CUTE/$f.py.ab_bak"
        cp "audit_out/graphreg/good_$f.py" "$CUTE/$f.py"
    done
    # snapshots are scratch-era: rewrite their intra-module imports to the promoted paths
    sed -i 's/from scratch\.gdn2_bwd_/from flash_mamba_rl.kernels.cute.gdn2_bwd_/' "$CUTE"/gdn2_bwd_*.py
    eval "$PP" CUDA_LAUNCH_BLOCKING=1 "$PY" scratch/gdn2_graph_bench.py \
        --shapes 1x512x4 --out "$OUTDIR/graphreg_ab_good.json" \
        > "$OUTDIR/graphreg_ab_good.log" 2>&1
    _rc "e(good-modules)" "$?"
    for f in gdn2_bwd_dhu gdn2_bwd_dhu_cw gdn2_bwd_wy gdn2_bwd_wy_cw; do
        mv "$CUTE/$f.py.ab_bak" "$CUTE/$f.py"
    done
    echo "  (modules restored)"
else
    _step "e: module A/B SKIPPED (capture passed pristine, or graphreg dir absent)"
fi

# ── f. full graph re-bench (4 shapes; incl. closed arm = Level-1a/1b numbers) ─
_step "f: full graph re-bench"
eval "$PP" "$PY" scratch/gdn2_graph_bench.py \
    --out "$OUTDIR/gdn2_graph_bench_v2.json" \
    > "$OUTDIR/gdn2_graph_bench_v2.log" 2>&1
_rc "f" "$?"

# ── g. Level-1b closed-path oracle gates ──────────────────────────────────────
_step "g: cw integration gate --closed (native grid, then L2048)"
eval "$PP" "$PY" scratch/gdn2_integration_box_cw.py --closed \
    --out "$OUTDIR/gdn2_integration_box_cw_closed.json" \
    > "$OUTDIR/gdn2_integration_box_cw_closed.log" 2>&1
_rc "g(grid)" "$?"
eval "$PP" "$PY" scratch/gdn2_integration_box_cw.py --closed --shapes "2,2048,8" \
    --out "$OUTDIR/gdn2_integration_box_cw_closed_L2048.json" \
    > "$OUTDIR/gdn2_integration_box_cw_closed_L2048.log" 2>&1
_rc "g(L2048)" "$?"

# ── h. Level-2 gate: epilogue-glue kernels ────────────────────────────────────
_step "h: k1_l2_epilogue --mode cw"
eval "$PP" "$PY" scratch/k1_l2_epilogue.py --mode cw \
    --out "$OUTDIR/k1_l2_epilogue.json" \
    > "$OUTDIR/k1_l2_epilogue.log" 2>&1
_rc "h" "$?"

# ── i. Level-3 gates: fused kernel on the offset-partition lifecycle ─────────
_step "i: k1_incb2_offsets (scalar, then cw) — separate processes"
for MODE in scalar cw; do
    eval "$PP" "$PY" scratch/k1_incb2_offsets.py --mode "$MODE" \
        --out "$OUTDIR/k1_incb2_v2_${MODE}.json" \
        > "$OUTDIR/k1_incb2_v2_${MODE}.log" 2>&1
    _rc "i(${MODE})" "$?"
done

# ── j. tiny GDN-2 training run (purchases "trains the family") ────────────────
_step "j: gdn2_tiny_train (native + eager comparison arms)"
if [ -f scratch/gdn2_tiny_train.py ]; then
    eval "$PP" "$PY" scratch/gdn2_tiny_train.py --arm native --steps 300 \
        --out "$OUTDIR/gdn2_tiny_train_native.json" \
        > "$OUTDIR/gdn2_tiny_train_native.log" 2>&1
    _rc "j(native)" "$?"
    eval "$PP" "$PY" scratch/gdn2_tiny_train.py --arm eager --steps 300 \
        --out "$OUTDIR/gdn2_tiny_train_eager.json" \
        > "$OUTDIR/gdn2_tiny_train_eager.log" 2>&1
    _rc "j(eager)" "$?"
    # fla arm intentionally not run: the mixer swap is unimplemented (would fabricate the curve)
else
    echo "  SKIP: scratch/gdn2_tiny_train.py not present"
fi

# ── k. fla Rosetta (parameterization mapping via naive_recurrent_gdn2) ────────
_step "k: fla_rosetta"
if [ -f scratch/fla_rosetta.py ]; then
    eval "$PP" "$PY" scratch/fla_rosetta.py \
        --out "$OUTDIR/fla_rosetta.json" \
        > "$OUTDIR/fla_rosetta.log" 2>&1
    _rc "k" "$?"
else
    echo "  SKIP: scratch/fla_rosetta.py not present"
fi

# ── l. family external parity rerun (SSD/KDA rows) ───────────────────────────
_step "l: family_external_parity rerun"
eval "$PP" "$PY" scratch/family_external_parity.py \
    --out "$OUTDIR/family_external_parity_v2.json" \
    > "$OUTDIR/family_external_parity_v2.log" 2>&1
_rc "l" "$?"

# ── m. mamba_ssm install + scan.cu re-measure ─────────────────────────────────
_step "m: mamba_ssm install + cuda_inc2_forward_bench"
bash scratch/install_mamba.sh > "$OUTDIR/install_mamba.log" 2>&1 || true
PYTHONPATH=src:. "$PY" -c 'import mamba_ssm; print("mamba_ssm available")' \
    > "$OUTDIR/mamba_check.log" 2>&1
if [ $? -eq 0 ]; then
    eval "$PP" "$PY" scratch/cuda_inc2_forward_bench.py > "$OUTDIR/inc2_bench.log" 2>&1
    _rc "m(inc2_bench)" "$?"
else
    echo "  SKIP: mamba_ssm not importable (see install_mamba.log)"
fi

# ── final env re-capture (proves the burst ended on the pristine .so) ─────────
_step "z: env re-capture"
_so_md5 > "$OUTDIR/so_md5_final.txt"
echo ""
echo "================================================================"
echo "  BURST-2 COMPLETE — output files:"
ls -la "$OUTDIR/"
echo "================================================================"
