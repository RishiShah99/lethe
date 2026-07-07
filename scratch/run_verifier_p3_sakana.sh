#!/usr/bin/env bash
# Box-side Phase-3: re-run the Sakana T4 differential with nvcc + ninja on PATH
# (Phase-2 failed to compile: the box has the CUDA runtime but not the toolkit
# on PATH). Reuses the manifest Phase-2 already built. Fresh shard files.
#
#   nohup bash scratch/run_verifier_p3_sakana.sh > ~/box_out_verifier/p3.log 2>&1 &

set -uo pipefail
OUTDIR=~/box_out_verifier
mkdir -p "$OUTDIR"
PY=~/cuteenv/bin/python
PP="PYTHONPATH=src:."
ALLCLASSES="matmul,attention,softmax,scan,norm,conv,reduction,elementwise,other"
export PATH="/usr/local/cuda/bin:$HOME/cuteenv/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="10.0"
cd ~/lethe || { echo "no ~/lethe"; exit 1; }

step() { echo ""; echo "================ $1 ================"; }
rc() { echo "  [$1] exit=$2"; }

step "nvcc / ninja sanity"
(nvcc --version | tail -2) 2>&1
(ninja --version) 2>&1

step "validate 2 rows (compile+audit) with toolkit on PATH"
timeout 900 bash -c "$PP $PY scratch/audit_extract_sakana.py --validate 2" \
  > "$OUTDIR/sakana_validate2.log" 2>&1
rc "validate" "$?"
tail -20 "$OUTDIR/sakana_validate2.log"

step "rebuild manifest with the fixed CUDA adapter (base64 + fwd-decl)"
"$PY" scratch/audit_extract_sakana.py --levels 1 --correct-only --limit 400 \
  scratch/audit_manifest_sakana.jsonl.gz > "$OUTDIR/sakana_manifest2.log" 2>&1
rc "manifest" "$?"
"$PY" -c "import gzip;print('manifest rows=',sum(1 for _ in gzip.open('scratch/audit_manifest_sakana.jsonl.gz','rt')))"

step "Sakana audit across 8 GPUs (fresh shards sakana2_*, timeout 400/row)"
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i nohup bash -c \
    "export PATH=/usr/local/cuda/bin:$HOME/cuteenv/bin:\$PATH; export CUDA_HOME=/usr/local/cuda; \
     $PP $PY scratch/audit_run.py scratch/audit_manifest_sakana.jsonl.gz \
     $OUTDIR/sakana2_shard$i.jsonl --device cuda --shard $i --num-shards 8 \
     --classes $ALLCLASSES --timeout 400" \
    > "$OUTDIR/sakana2_audit_$i.log" 2>&1 &
done
wait

step "aggregate Sakana native differential (glob quoted, one arg)"
cat "$OUTDIR"/sakana2_shard*.jsonl > "$OUTDIR/sakana2_all.jsonl" 2>/dev/null
PYTHONPATH=src:. "$PY" scratch/audit_aggregate_sakana.py "$OUTDIR/sakana2_all.jsonl" \
  scratch/audit_manifest_sakana.jsonl.gz --json "$OUTDIR/audit_sakana.json" \
  > "$OUTDIR/sakana_agg2.log" 2>&1
rc "aggregate" "$?"
tail -30 "$OUTDIR/sakana_agg2.log"

step "DONE"
echo "p3 complete"
