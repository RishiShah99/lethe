#!/usr/bin/env bash
# inc-B2 launch-SIGSEGV — round 4. THE FIX: range_constexpr UNROLLS the reverse loop (no scf.for),
# so the tcgen05 epilogue copy atom lives only in straight-line code. Separate process per variant.
#   TILE dyn       → GO   (clean proven body, in-loop everything, outer loop now unrolled)
#   BISECT 1 NT1   → GO   (real kernel, G1-only)
#   BISECT 0 NT1   → GO   (full fused chunk)
#   BISECT 0 NT4   → GO   (full + the reverse b_dh carry across 4 unrolled chunks)
set +e
cd ~/lethe || exit 2
export PYTHONPATH=src:.
PY=~/cuteenv/bin/python

echo "===== gen bundles ====="
$PY scratch/gen_k1_bundle.py --nt 1 --out k1_bundle_nt1.pt
$PY scratch/gen_k1_bundle.py --nt 4 --out k1_bundle_nt4.pt

echo "===== TILE dyn (proven body, unrolled, L=4) ====="
$PY scratch/loop_tile_repro.py --mode dyn --L 4 --out results/tile_dyn_unroll.json
echo "EXIT_TILE_dyn=$?"

echo "===== BISECT 1 (incB2 G1-only, unrolled, NT=1) ====="
$PY scratch/k1_microgate.py --mode incB2 --bisect 1 --bundle k1_bundle_nt1.pt \
    --out results/k1_microgate_bisect1.json
echo "EXIT_BISECT_1=$?"

echo "===== BISECT 0 (incB2 full, unrolled, NT=1) ====="
$PY scratch/k1_microgate.py --mode incB2 --bisect 0 --bundle k1_bundle_nt1.pt \
    --out results/k1_microgate_bisect0_nt1.json
echo "EXIT_BISECT_0_NT1=$?"

echo "===== BISECT 0 (incB2 full, unrolled, NT=4 — the reverse carry) ====="
$PY scratch/k1_microgate.py --mode incB2 --bisect 0 --bundle k1_bundle_nt4.pt \
    --out results/k1_microgate_bisect0_nt4.json
echo "EXIT_BISECT_0_NT4=$?"

echo "ALL_DONE_FIX4"
