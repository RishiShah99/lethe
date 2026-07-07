#!/usr/bin/env bash
# Box-side Phase-2: T2 positive control + T4 Sakana second-corpus differential.
# Run AFTER run_verifier_p1.sh has finished (needs the GPUs free). Results in
# ~/box_out_verifier. Sakana's CUDA path (load_inline) may hit toolchain drift;
# the aggregator excludes compile failures, so a low compiled_denominator is
# self-reporting rather than fatal.
#
#   nohup bash scratch/run_verifier_p2.sh > ~/box_out_verifier/p2.log 2>&1 &

set -uo pipefail
OUTDIR=~/box_out_verifier
mkdir -p "$OUTDIR"
PY=~/cuteenv/bin/python
UV=~/.local/bin/uv
PP="PYTHONPATH=src:."
ALLCLASSES="matmul,attention,softmax,scan,norm,conv,reduction,elementwise,other"
export TORCH_CUDA_ARCH_LIST="10.0"
cd ~/lethe || { echo "no ~/lethe"; exit 1; }

step() { echo ""; echo "================ $1 ================"; }
rc() { echo "  [$1] exit=$2"; }

step "T2: positive control — our kernels through the SAME audit battery"
eval "$PP" "$PY" scratch/positive_control.py --device cuda \
  --json "$OUTDIR/positive_control.json" > "$OUTDIR/positive_control.log" 2>&1
rc "T2" "$?"
tail -20 "$OUTDIR/positive_control.log"

step "T4-prep: ensure ninja for load_inline; check nvcc"
"$PY" -c "import ninja" 2>/dev/null || "$UV" pip install --python "$PY" ninja > "$OUTDIR/ninja_install.log" 2>&1
(nvcc --version || echo "NO nvcc in PATH") > "$OUTDIR/nvcc.txt" 2>&1
tail -3 "$OUTDIR/nvcc.txt"

step "T4 Sakana: adapter validate (compile+audit 2 Correct rows)"
timeout 900 bash -c "$PP $PY scratch/audit_extract_sakana.py --validate 2" \
  > "$OUTDIR/sakana_validate.log" 2>&1
rc "validate" "$?"
tail -25 "$OUTDIR/sakana_validate.log"

step "T4 Sakana: build manifest (level 1, Correct=True, cap 400)"
"$PY" scratch/audit_extract_sakana.py --levels 1 --correct-only --limit 400 \
  scratch/audit_manifest_sakana.jsonl.gz > "$OUTDIR/sakana_manifest.log" 2>&1
rc "manifest" "$?"
tail -3 "$OUTDIR/sakana_manifest.log"

step "T4 Sakana: audit across 8 GPUs (load_inline compile per row; timeout 400)"
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i nohup bash -c \
    "$PP $PY scratch/audit_run.py scratch/audit_manifest_sakana.jsonl.gz \
     $OUTDIR/sakana_shard$i.jsonl --device cuda --shard $i --num-shards 8 \
     --classes $ALLCLASSES --timeout 400" \
    > "$OUTDIR/sakana_audit_$i.log" 2>&1 &
done
wait

step "T4 Sakana: aggregate the native differential"
eval "$PP" "$PY" scratch/audit_aggregate_sakana.py "$OUTDIR/sakana_shard*.jsonl" \
  scratch/audit_manifest_sakana.jsonl.gz --json "$OUTDIR/audit_sakana.json" \
  > "$OUTDIR/sakana_agg.log" 2>&1
rc "aggregate" "$?"
tail -30 "$OUTDIR/sakana_agg.log"

step "T4 Sakana: external KernelBench check on the SAME Sakana kernels (2nd cell)"
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i nohup bash -c \
    "$PP $PY scratch/external_kernelbench_check.py \
     scratch/audit_manifest_sakana.jsonl.gz $OUTDIR/sakana_crossval_$i.jsonl \
     --device cuda --shard $i --num-shards 8 --classes $ALLCLASSES --timeout 400" \
    > "$OUTDIR/sakana_crossval_log_$i.log" 2>&1 &
done
wait
cat "$OUTDIR"/sakana_crossval_*.jsonl > "$OUTDIR/sakana_crossval_rows.jsonl" 2>/dev/null

step "DONE — pull ~/box_out_verifier"
echo "p2 complete"
