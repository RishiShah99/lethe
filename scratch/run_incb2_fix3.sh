#!/usr/bin/env bash
# inc-B2 launch-SIGSEGV — round 3. Find the MINIMAL hoist on the clean proven body.
#   dyn   = control (everything in-loop, dynamic-L coord) — EXPECT 139 (confirms build still faults)
#   hoist = candidate fix: all make_* atoms/fragments hoisted OUTSIDE the loop, views in-loop.
#           GO ⇒ the launch fault is the in-loop make_* creation; hoisting them fixes inc-B2.
#           139 ⇒ a view op (local_tile/tma_partition) must hoist too (keep-L-mode) — escalate.
set +e
cd ~/flash-mamba-rl || exit 2
export PYTHONPATH=src:.
PY=~/cuteenv/bin/python

for M in dyn hoist; do
  echo "===== TILE ${M} (proven body, L=4) ====="
  $PY scratch/loop_tile_repro.py --mode ${M} --L 4 --out results/tile_${M}.json
  echo "EXIT_TILE_${M}=$?"
done

echo "ALL_DONE_FIX3"
