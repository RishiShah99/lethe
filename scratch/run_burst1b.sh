#!/usr/bin/env bash
# Burst-1b: the controlled #3259 .so experiment + burst-1 leftovers.
# Swap -libs-base -> -libs-cu13 (same pinned 4.5.2) and rerun every env-sensitive
# fault-matrix leg on the SAME box, same day; plus cuLA install retry, mamba_ssm
# hunt, and the bench-scale (L=2048) integration gate.
set -o pipefail
OUTDIR=~/box_out_burst1
mkdir -p "$OUTDIR"
PY=~/cuteenv/bin/python
UV=~/.local/bin/uv

echo "== md5 BEFORE (only -libs-base installed) =="
find ~/cuteenv/lib/python3.12/site-packages/ -name 'libcute_dsl_runtime.so' -exec md5sum {} \; \
    | tee "$OUTDIR/so_md5_before_cu13.txt"

echo "== install -libs-cu13==4.5.2 =="
"$UV" pip install --python "$PY" nvidia-cutlass-dsl-libs-cu13==4.5.2 \
    > "$OUTDIR/cu13_install.log" 2>&1
echo "install_rc=$?" | tee -a "$OUTDIR/cu13_install.log"

echo "== md5 AFTER =="
find ~/cuteenv/lib/python3.12/site-packages/ -name 'libcute_dsl_runtime.so' -exec md5sum {} \; \
    | tee "$OUTDIR/so_md5_after_cu13.txt"

cd ~/flash-mamba-rl || exit 1

echo "== rerun fault-matrix legs under cu13 =="
for MODE in straight loop loopfix straight2; do
    PYTHONPATH=src:. "$PY" scratch/loop_gemm_repro.py --mode "$MODE" \
        --out "$OUTDIR/cu13_loop_gemm_${MODE}.json" \
        > "$OUTDIR/cu13_loop_gemm_${MODE}.log" 2>&1
    echo "  loop_gemm(${MODE}) rc=$?"
done
for MODE in static dyn hoist; do
    PYTHONPATH=src:. "$PY" scratch/loop_tile_repro.py --mode "$MODE" --L 4 \
        --out "$OUTDIR/cu13_loop_tile_${MODE}.json" \
        > "$OUTDIR/cu13_loop_tile_${MODE}.log" 2>&1
    echo "  loop_tile(${MODE}) rc=$?"
done
for MODE in independent looped dependent; do
    PYTHONPATH=src:. "$PY" scratch/tmem_offset_probe.py --mode "$MODE" \
        --out "$OUTDIR/cu13_tmem_probe_${MODE}.json" \
        > "$OUTDIR/cu13_tmem_probe_${MODE}.log" 2>&1
    echo "  tmem_probe(${MODE}) rc=$?"
done

echo "== integration gate under cu13 (safety: production kernels must stay GO) =="
PYTHONPATH=src:. "$PY" scratch/gdn2_integration_box_cw.py \
    --out "$OUTDIR/cu13_integration_cw.json" \
    > "$OUTDIR/cu13_integration_cw.log" 2>&1
echo "  integration rc=$?"

echo "== bench-scale integration gate (L=2048) =="
PYTHONPATH=src:. "$PY" scratch/gdn2_integration_box_cw.py \
    --shapes "2,2048,8" \
    --out "$OUTDIR/cu13_integration_cw_L2048.json" \
    > "$OUTDIR/cu13_integration_cw_L2048.log" 2>&1
echo "  integration_L2048 rc=$?"

echo "== family gates under cu13 =="
PYTHONPATH=src:. "$PY" scratch/gdn2_family_box.py \
    --out "$OUTDIR/cu13_gdn2_family_box.json" \
    > "$OUTDIR/cu13_gdn2_family_box.log" 2>&1
echo "  family rc=$?"

echo "== cuLA retry with CUDA_HOME =="
CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH \
    "$UV" pip install --python "$PY" \
    'git+https://github.com/inclusionAI/cuLA@6cacc37fd420e72c859c1dec6c870a13dc2a0e9a' \
    > "$OUTDIR/cula_install2.log" 2>&1
echo "  cula_rc=$?" | tee -a "$OUTDIR/cula_install2.log"

echo "== mamba_ssm hunt (for the scan.cu re-measure) =="
{
    ls -d ~/*env* 2>/dev/null
    for E in ~/*env*/bin/python; do
        echo "--- $E"
        "$E" -c "import mamba_ssm; print('mamba_ssm', mamba_ssm.__version__)" 2>&1 | tail -1
    done
} | tee "$OUTDIR/mamba_ssm_hunt.txt"

echo "== BURST-1B DONE =="
ls -la "$OUTDIR/" | grep cu13
