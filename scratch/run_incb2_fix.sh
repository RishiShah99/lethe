#!/usr/bin/env bash
# inc-B2 launch-SIGSEGV FIX burst — the in-loop TMEM-handle idiom.
#
# Each variant is a SEPARATE python process so a segfault in one cannot poison the next
# process's fresh CUDA context. set +e: keep going past a crash; the EXIT_* lines are the
# signal (139 = SIGSEGV, no JSON written; 0 = ran, then read GO from the JSON).
#
# Decisive reads:
#   REPRO straight  → GO  (control; if it crashes, the transcription is wrong, discard the run)
#   REPRO loop      → 139 expected (loop-wrapping the proven body with the handle captured)
#   REPRO loopfix   → GO  ⇒ the in-loop TMEM-handle idiom is the fix
#   BISECT 5 (NT=1) → GO  ⇒ the fix clears the real kernel's G1-only minimal path
#   BISECT 6 (NT=1) → GO  ⇒ the full fused chunk (G1+GA+round-trips) clears at NT=1
#   BISECT 6 (NT=4) → GO  ⇒ the carry across 4 reverse iterations is correct
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

for M in straight loop loopfix; do
  echo "===== REPRO ${M} (proven GEMM, decoupled) ====="
  $PY scratch/loop_gemm_repro.py --mode ${M} --out results/repro_${M}.json
  echo "EXIT_REPRO_${M}=$?"
done

echo "===== BISECT 5 (incB2 G1-only + in-loop handle, NT=1) ====="
$PY scratch/k1_microgate.py --mode incB2 --bisect 5 --bundle k1_bundle_nt1.pt \
    --out results/k1_microgate_bisect5.json
echo "EXIT_BISECT_5=$?"

echo "===== BISECT 6 (incB2 full + in-loop handle, NT=1) ====="
$PY scratch/k1_microgate.py --mode incB2 --bisect 6 --bundle k1_bundle_nt1.pt \
    --out results/k1_microgate_bisect6_nt1.json
echo "EXIT_BISECT_6_NT1=$?"

echo "===== BISECT 6 (incB2 full + in-loop handle, NT=4 — the reverse carry) ====="
$PY scratch/k1_microgate.py --mode incB2 --bisect 6 --bundle k1_bundle_nt4.pt \
    --out results/k1_microgate_bisect6_nt4.json
echo "EXIT_BISECT_6_NT4=$?"

echo "ALL_DONE_FIX"
