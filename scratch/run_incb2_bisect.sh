#!/usr/bin/env bash
# inc-B2 launch-SIGSEGV bisection burst. Each variant is a SEPARATE python process so a
# segfault in one cannot poison the next process's fresh CUDA context. set +e: keep going.
set +e
cd ~/flash-mamba-rl || exit 2
export PYTHONPATH=src:.
PY=~/cuteenv/bin/python

echo "===== gen bundles ====="
$PY scratch/gen_k1_bundle.py --nt 1 --out k1_bundle_nt1.pt
$PY scratch/gen_k1_bundle.py --nt 4 --out k1_bundle_nt4.pt

echo "===== SANITY: lever-B batched (silicon-proven) NT=1 ====="
$PY scratch/k1_microgate.py --mode incB --bundle k1_bundle_nt1.pt --out results/k1_sanity_incB.json
echo "EXIT_SANITY=$?"

for B in 1 2 0; do
  echo "===== BISECT ${B} (incB2, NT=1) ====="
  $PY scratch/k1_microgate.py --mode incB2 --bisect ${B} --bundle k1_bundle_nt1.pt \
      --out results/k1_microgate_bisect${B}.json
  echo "EXIT_BISECT_${B}=$?"
done

echo "ALL_DONE"
