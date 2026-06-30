#!/usr/bin/env bash
# inc-B2 bisection round 3 — the two suspect-#3 / relinquish probes that round 2 failed to run.
set +e
cd ~/flash-mamba-rl || exit 2
export PYTHONPATH=src:.
PY=~/cuteenv/bin/python

for B in 3 4; do
  echo "===== BISECT ${B} (incB2, NT=1) ====="
  $PY scratch/k1_microgate.py --mode incB2 --bisect ${B} --bundle k1_bundle_nt1.pt \
      --out results/k1_microgate_bisect${B}.json
  echo "EXIT_BISECT_${B}=$?"
done

echo "ALL_DONE3"
