#!/usr/bin/env bash
# cuda-gdb backtrace on the minimal inc-B2 repro (bisect=1) — localize the launch SIGSEGV
# (host vs device, and which frame). memcheck on catches device illegal accesses with a PC.
set +e
cd ~/lethe || exit 2
export PYTHONPATH=src:.
PY=~/cuteenv/bin/python
GDB=/usr/local/cuda/bin/cuda-gdb

echo "GDB=$($GDB --version 2>&1 | head -1)"
$GDB -q -batch \
    -ex "set cuda memcheck on" \
    -ex "set cuda api_failures stop" \
    -ex "run" \
    -ex "bt" \
    -ex "info cuda kernels" \
    --args $PY scratch/k1_microgate.py --mode incB2 --bisect 1 \
    --bundle k1_bundle_nt1.pt --out results/gdb_b1.json 2>&1
echo "GDB_DONE"
