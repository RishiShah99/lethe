#!/usr/bin/env bash
# Localize the inc-B2 launch SIGSEGV on the minimal repro (bisect=1, G1-only).
set +e
cd ~/flash-mamba-rl || exit 2
export PYTHONPATH=src:.
PY=~/cuteenv/bin/python
SAN=/usr/local/cuda/bin/compute-sanitizer

echo "===== env ====="
which gdb
$SAN --version 2>&1 | head -4
nvidia-smi --query-gpu=driver_version --format=csv,noheader

echo "===== plain bisect=1, CUDA_LAUNCH_BLOCKING=1 (synchronous device errors) ====="
CUDA_LAUNCH_BLOCKING=1 $PY scratch/k1_microgate.py --mode incB2 --bisect 1 \
    --bundle k1_bundle_nt1.pt --out results/lb_b1.json 2>&1
echo "LB_EXIT=$?"

echo "===== compute-sanitizer memcheck bisect=1 ====="
$SAN --tool memcheck --target-processes all \
    $PY scratch/k1_microgate.py --mode incB2 --bisect 1 \
    --bundle k1_bundle_nt1.pt --out results/sanit_b1.json 2>&1
echo "SAN_EXIT=$?"

echo "ALL_DIAG_DONE"
