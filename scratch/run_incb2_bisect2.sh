#!/usr/bin/env bash
# inc-B2 bisection round 2 — isolate suspect #3 (loop-induction TMA coord) and relinquish order.
# bisect=1 baseline (known SIGSEGV), 3 = G1 with bh (not arithmetic lid) coord, 4 = relinquish
# after mainloop. Each a separate process.
set +e
cd ~/lethe || exit 2
export PYTHONPATH=src:.
PY=~/cuteenv/bin/python

for B in 1 3 4; do
  echo "===== BISECT ${B} (incB2, NT=1) ====="
  $PY scratch/k1_microgate.py --mode incB2 --bisect ${B} --bundle k1_bundle_nt1.pt \
      --out results/k1_microgate_bisect${B}.json
  echo "EXIT_BISECT_${B}=$?"
done

echo "ALL_DONE2"
