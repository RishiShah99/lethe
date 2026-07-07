#!/usr/bin/env bash
# inc-B2 launch-SIGSEGV — round 2. The handle is now bound IN-LOOP (loopfix idiom); this burst
# (a) isolates the in-loop dynamic-L local_tile/tma_partition on the clean proven body, and
# (b) re-runs the now-compile-fixed inc-B2 G1-only / full paths.  Separate process per variant.
#
# Decisive reads:
#   TILE static → GO       (in-loop tile/TMA derivation with a constant L coord)
#   TILE dyn    → GO/139   (the EXACT inc-B2 G1 pattern: in-loop tile/TMA with the loop var as
#                           the L coord) — 139 ⇒ the dynamic-L TMA coord is the launch fault
#   BISECT 1    → GO/139   (real kernel, G1-only, in-loop handle) — compile must now succeed
#   BISECT 0 NT1/NT4 → GO/139  (full fused chunk + reverse carry)
set +e
cd ~/lethe || exit 2
export PYTHONPATH=src:.
PY=~/cuteenv/bin/python

echo "===== gen bundles ====="
$PY scratch/gen_k1_bundle.py --nt 1 --out k1_bundle_nt1.pt
$PY scratch/gen_k1_bundle.py --nt 4 --out k1_bundle_nt4.pt

for M in static dyn; do
  echo "===== TILE ${M} (proven body, in-loop tile/TMA, L=4) ====="
  $PY scratch/loop_tile_repro.py --mode ${M} --L 4 --out results/tile_${M}.json
  echo "EXIT_TILE_${M}=$?"
done

echo "===== BISECT 1 (incB2 G1-only, in-loop handle, NT=1) ====="
$PY scratch/k1_microgate.py --mode incB2 --bisect 1 --bundle k1_bundle_nt1.pt \
    --out results/k1_microgate_bisect1.json
echo "EXIT_BISECT_1=$?"

echo "===== BISECT 0 (incB2 full, in-loop handle, NT=1) ====="
$PY scratch/k1_microgate.py --mode incB2 --bisect 0 --bundle k1_bundle_nt1.pt \
    --out results/k1_microgate_bisect0_nt1.json
echo "EXIT_BISECT_0_NT1=$?"

echo "===== BISECT 0 (incB2 full, in-loop handle, NT=4 — the reverse carry) ====="
$PY scratch/k1_microgate.py --mode incB2 --bisect 0 --bundle k1_bundle_nt4.pt \
    --out results/k1_microgate_bisect0_nt4.json
echo "EXIT_BISECT_0_NT4=$?"

echo "ALL_DONE_FIX2"
