#!/usr/bin/env bash
# Box-side Phase-1 of the verifier-evidence session (T1.3 + T3-exec).
# Deterministic manifest regen, then reconfirm + the differential cross-val.
# Result JSON/logs in ~/box_out_verifier. Each step writes incrementally so an
# SSH drop mid-run loses only the running step.
#
# Usage (detached via scratch/launch_verifier_p1.sh):
#   nohup bash scratch/run_verifier_p1.sh > ~/box_out_verifier/p1.log 2>&1 &

set -uo pipefail
OUTDIR=~/box_out_verifier
mkdir -p "$OUTDIR"
PY=~/cuteenv/bin/python
UV=~/.local/bin/uv
PP="PYTHONPATH=src:."
cd ~/lethe || { echo "no ~/lethe"; exit 1; }

step() { echo ""; echo "================ $1 ================"; }
rc() { echo "  [$1] exit=$2"; }

step "a: deps (datasets + pyarrow on cuteenv)"
"$PY" -c "import datasets, pyarrow" 2>/dev/null \
  || "$UV" pip install --python "$PY" "datasets>=2.0" pyarrow huggingface_hub \
       > "$OUTDIR/dep_install.log" 2>&1
"$PY" -c "import datasets,pyarrow,torch;print('deps ok datasets',datasets.__version__,'torch',torch.__version__)" \
  > "$OUTDIR/deps.txt" 2>&1
cat "$OUTDIR/deps.txt"

step "b: regenerate Dr.Kernel manifest if absent"
if [ ! -f scratch/audit_manifest_drkernel.jsonl.gz ]; then
  HF_HUB_DISABLE_SYMLINKS_WARNING=1 "$PY" - <<'PYEOF' > "$OUTDIR/manifest_regen.log" 2>&1
from datasets import load_dataset
ds = load_dataset("hkust-nlp/drkernel-coldstart-8k", split="train")
ds.to_parquet("scratch/drkernel_coldstart.parquet")
print("saved parquet rows=", ds.num_rows)
PYEOF
  eval "$PP" "$PY" scratch/audit_extract_drkernel.py \
    scratch/drkernel_coldstart.parquet scratch/audit_manifest_drkernel.jsonl.gz \
    >> "$OUTDIR/manifest_regen.log" 2>&1
  rc "b-extract" "$?"
fi
"$PY" -c "import gzip,json;print('manifest rows=',sum(1 for _ in gzip.open('scratch/audit_manifest_drkernel.jsonl.gz','rt')))" \
  | tee -a "$OUTDIR/manifest_regen.log"

step "c: T1.3 reconfirm — 300-row sample audit on GPU0, compare finding rate"
CUDA_VISIBLE_DEVICES=0 timeout 3600 bash -c \
  "$PP $PY scratch/audit_run.py scratch/audit_manifest_drkernel.jsonl.gz \
   $OUTDIR/reconfirm_sample.jsonl --device cuda --shard 0 --num-shards 8 --limit 300" \
  > "$OUTDIR/reconfirm.log" 2>&1
rc "c" "$?"
eval "$PP" "$PY" scratch/audit_aggregate.py "$OUTDIR/reconfirm_sample.jsonl" \
  --json "$OUTDIR/reconfirm_agg.json" > "$OUTDIR/reconfirm_agg.log" 2>&1
tail -6 "$OUTDIR/reconfirm_agg.log"

step "d: T3-exec — external KernelBench check over accepted rows, 8 shards"
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i nohup bash -c \
    "$PP $PY scratch/external_kernelbench_check.py \
     scratch/audit_manifest_drkernel.jsonl.gz $OUTDIR/crossval_rows_$i.jsonl \
     --device cuda --shard $i --num-shards 8 --only-accepted --timeout 120" \
    > "$OUTDIR/crossval_$i.log" 2>&1 &
done
wait
cat "$OUTDIR"/crossval_rows_*.jsonl > "$OUTDIR/crossval_rows.jsonl" 2>/dev/null
"$PY" -c "print('crossval rows=',sum(1 for _ in open('$OUTDIR/crossval_rows.jsonl')))" \
  | tee -a "$OUTDIR/p1.log"
"$PY" - <<'PYEOF' | tee -a "$OUTDIR/p1.log"
import json, glob
n=p=0
for f in glob.glob(__import__('os').path.expanduser('~/box_out_verifier/crossval_rows_*.jsonl')):
    for l in open(f):
        r=json.loads(l); pe=r.get('paper_era') or {}
        n+=1; p+= 1 if pe.get('correct') else 0
print(f"external paper-era: {p}/{n} PASS ({p/max(1,n):.1%})")
PYEOF

step "DONE — pull ~/box_out_verifier (join runs locally vs full cached shards)"
echo "p1 complete"
