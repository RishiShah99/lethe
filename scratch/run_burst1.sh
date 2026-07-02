#!/usr/bin/env bash
# Box-side burst-1 orchestration.  Runs from the repo root on the B200 box.
# Each step is independent (|| true); status is captured and echoed.
# All JSON output lands in ~/box_out_burst1/.
#
# Usage:
#   nohup bash scratch/run_burst1.sh > ~/box_out_burst1/burst1.log 2>&1 &
#
# The script assumes:
#   - cuteenv at ~/cuteenv/bin/python (Python 3.12, torch 2.11, triton 3.6, fla 0.5.1, cutlass-dsl 4.5.2)
#   - PYTHONPATH set to src:. (repo root is cwd)
#   - CUDA_HOME=/usr/local/cuda-13.0 (CCCL headers; already provisioned)

set -o pipefail
OUTDIR=~/box_out_burst1
mkdir -p "$OUTDIR"
PY=~/cuteenv/bin/python
PP=PYTHONPATH=src:.
CUDA_PREFIX="CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH"

_step() {
    local label="$1"
    echo ""
    echo "================================================================"
    echo "  STEP: $label"
    echo "================================================================"
}

_rc() {
    local label="$1"
    local code="$2"
    echo "  [${label}] exit_code=${code}"
}

# ── a. Env capture ──────────────────────────────────────────────────────────
_step "a: env capture"
nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader \
    > "$OUTDIR/nvidia_smi.txt" 2>&1
_RC_A=$?

eval "$PP" "$PY" - <<'PYEOF' > "$OUTDIR/env_burst1.json" 2>&1
import json, torch, sys

def _ver(mod_name):
    try:
        import importlib; m = importlib.import_module(mod_name)
        return getattr(m, "__version__", "present_no_version")
    except ImportError:
        return "not_importable"

props = {}
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    props = {
        "name": p.name,
        "sm_count": p.multi_processor_count,
        "total_memory_gb": p.total_memory / 1e9,
        "compute_capability": f"{p.major}.{p.minor}",
    }

info = {
    "python": sys.version,
    "torch": torch.__version__,
    "triton": _ver("triton"),
    "fla": _ver("fla"),
    "cutlass_dsl": _ver("cutlass"),
    "cuda_available": torch.cuda.is_available(),
    "device_props": props,
}
print(json.dumps(info, indent=2))
PYEOF
_RC_A2=$?
_rc "a(nvidia-smi)" "$_RC_A"
_rc "a(env_json)" "$_RC_A2"

# ── b. #3259 .so md5 check ──────────────────────────────────────────────────
_step "b: libcute_dsl_runtime.so md5 + pip show"
SO_PATH=$(find ~/cuteenv/lib/python3.12/site-packages/ -name 'libcute_dsl_runtime.so' 2>/dev/null | head -1)
if [ -n "$SO_PATH" ]; then
    md5sum "$SO_PATH" > "$OUTDIR/so_check.txt" 2>&1
    echo "so_path=$SO_PATH" >> "$OUTDIR/so_check.txt"
else
    echo "libcute_dsl_runtime.so NOT FOUND" > "$OUTDIR/so_check.txt"
fi
~/cuteenv/bin/pip show nvidia-cutlass-dsl-libs-base nvidia-cutlass-dsl-libs-cu13 \
    >> "$OUTDIR/so_check.txt" 2>&1 || true
_rc "b" "$?"

# ── c. loop_gemm_repro.py — four modes, separate processes ──────────────────
_step "c: loop_gemm_repro (straight / loop / loopfix / straight2)"
for MODE in straight loop loopfix straight2; do
    echo "  --- loop_gemm_repro --mode $MODE ---"
    eval "$PP" "$PY" scratch/loop_gemm_repro.py \
        --mode "$MODE" \
        --out "$OUTDIR/loop_gemm_repro_${MODE}.json" \
        > "$OUTDIR/loop_gemm_repro_${MODE}.log" 2>&1
    RC=$?
    _rc "c(${MODE})" "$RC"
done

# ── d. loop_tile_repro.py — static and dyn modes (no hoist: needs dynamic L) ─
_step "d: loop_tile_repro (static / dyn / hoist)"
for MODE in static dyn hoist; do
    echo "  --- loop_tile_repro --mode $MODE --L 4 ---"
    eval "$PP" "$PY" scratch/loop_tile_repro.py \
        --mode "$MODE" --L 4 \
        --out "$OUTDIR/loop_tile_repro_${MODE}.json" \
        > "$OUTDIR/loop_tile_repro_${MODE}.log" 2>&1
    RC=$?
    _rc "d(${MODE})" "$RC"
done

# ── e. tmem_offset_probe.py — three modes ───────────────────────────────────
# NOTE: tmem_offset_probe.py is authored in a separate session task.
# Wire the call now; if the file doesn't exist yet, the step records the error.
_step "e: tmem_offset_probe (independent / looped / dependent)"
if [ -f scratch/tmem_offset_probe.py ]; then
    for MODE in independent looped dependent; do
        echo "  --- tmem_offset_probe --mode $MODE ---"
        eval "$PP" "$PY" scratch/tmem_offset_probe.py \
            --mode "$MODE" \
            --out "$OUTDIR/tmem_offset_probe_${MODE}.json" \
            > "$OUTDIR/tmem_offset_probe_${MODE}.log" 2>&1
        RC=$?
        _rc "e(${MODE})" "$RC"
    done
else
    echo "  SKIP: scratch/tmem_offset_probe.py not present (authored separately)" \
        > "$OUTDIR/tmem_offset_probe_skip.txt"
    _rc "e" "0 (skipped)"
fi

# ── f. fla_gdn2_probe.py ────────────────────────────────────────────────────
_step "f: fla_gdn2_probe"
eval "$PP" "$PY" scratch/fla_gdn2_probe.py \
    --out "$OUTDIR/fla_gdn2_probe.json" \
    > "$OUTDIR/fla_gdn2_probe.log" 2>&1
RC=$?
_rc "f(fla_gdn2_probe)" "$RC"

# ── g. gdn2_family_box.py ────────────────────────────────────────────────────
_step "g: gdn2_family_box"
eval "$PP" "$PY" scratch/gdn2_family_box.py \
    --out "$OUTDIR/gdn2_family_box.json" \
    > "$OUTDIR/gdn2_family_box.log" 2>&1
RC=$?
_rc "g(gdn2_family_box)" "$RC"

# ── h. family_external_parity.py (after best-effort cula install) ────────────
_step "h: family_external_parity (+ best-effort cula install)"
# cuLA = inclusionAI/cuLA, pinned at the recon HEAD 6cacc37 (frontier_pins).
~/.local/bin/uv pip install --python ~/cuteenv/bin/python \
    'git+https://github.com/inclusionAI/cuLA@6cacc37fd420e72c859c1dec6c870a13dc2a0e9a' \
    > "$OUTDIR/cula_install.log" 2>&1 || true
echo "cula install exit=$?" >> "$OUTDIR/cula_install.log"

eval "$PP" "$PY" scratch/family_external_parity.py \
    --out "$OUTDIR/family_external_parity.json" \
    > "$OUTDIR/family_external_parity.log" 2>&1
RC=$?
_rc "h(family_external_parity)" "$RC"

# ── i. Integration gates: gdn2_integration_box_cw.py (native shapes) ─────────
_step "i: gdn2_integration_box_cw (native shapes)"
eval "$PP" "$PY" scratch/gdn2_integration_box_cw.py \
    --out "$OUTDIR/gdn2_integration_box_cw.json" \
    > "$OUTDIR/gdn2_integration_box_cw.log" 2>&1
RC=$?
_rc "i(cw_gate)" "$RC"

# gdn2_graph.py is a library module (no __main__/argparse); skip as standalone.
# The integration gate above already exercises the assembled backward path.
echo "  NOTE: scratch/gdn2_graph.py is a library module — no standalone run wired."

# ── j. cuda_inc2_forward_bench.py (mamba_ssm guard) ─────────────────────────
_step "j: cuda_inc2_forward_bench (mamba_ssm importable guard)"
eval "$PP" "$PY" -c "import mamba_ssm; print('mamba_ssm available')" \
    > "$OUTDIR/inc2_bench_mamba_check.log" 2>&1
RC_MAMBA=$?
if [ "$RC_MAMBA" -eq 0 ]; then
    eval "$PP" "$PY" scratch/cuda_inc2_forward_bench.py \
        > "$OUTDIR/inc2_bench.log" 2>&1
    _rc "j(inc2_bench)" "$?"
else
    echo "  SKIP: mamba_ssm not importable" >> "$OUTDIR/inc2_bench_mamba_check.log"
    _rc "j(mamba_ssm_not_found)" "0 (skipped)"
fi

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  BURST-1 COMPLETE — output files:"
ls -la "$OUTDIR/"
echo "================================================================"
