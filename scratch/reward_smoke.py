"""GPU smoke for the hardened + continuous reward path (reward-integrity burst).

Re-scores SFT warm-start targets through the EXACT GRPO scoring path with
speedup measurement ON, to confirm on-device that the reward-integrity work
behaves correctly on *known-correct* candidates (the risk being a new screen
or gate that falsely breaks a real kernel):

  - M2 continuous reward: a parity triton target scores ~1.0 smoothly (not the
    old bimodal 0.5/1.0 cliff); a slow eager target stays pinned at the 0.5
    floor. The eager/triton split is the on-device evidence the band exists.
  - C3 bench-shape correctness gate fires (``correct_at_bench`` reported; a
    mismatch would demote to the 0.1 contract-fail reward).
  - C2 per-trial inputs run (``measure_speedup`` completes; speedup is honest,
    not a memoization/in-place artifact).
  - C1 AST screen + M1 ORD-02 pass the self-contained triton targets on-device.
  - bug-routing flag is True for the #904-casualty ops on sm_100 (bonus only
    pays when speedup > 1.0, so parity targets correctly earn no bonus).

Usage (box):
    uv run python scratch/reward_smoke.py --variants triton --ops forward_chunked_scan,complex_scan_rope,fused_block_forward
    uv run python scratch/reward_smoke.py --variants eager --ops forward_chunked_scan
    uv run python scratch/reward_smoke.py --variants triton --out reward_smoke_triton.json
"""

from __future__ import annotations

import argparse
import json
import sys

from flash_mamba_rl.rl.sft_targets import available_targets, target_source, target_variants
from flash_mamba_rl.verifier.candidate_scoring import (
    DEFAULT_EXCLUDE_GATES,
    score_candidate_source,
)

# Mirrors sft_validate.TIMEOUT_S; speedup adds one bench pass (warmup+20 trials
# at the 1024-wide bench shape) on top of the gate battery.
TIMEOUT_S: dict[str, float] = {
    "forward_chunked_scan": 900.0,
    "complex_scan_rope": 900.0,
    "fused_block_forward": 1400.0,
    "backward_selective_scan": 2000.0,
    "mimo_backward": 2200.0,
    "fused_block_backward": 3000.0,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ops", default="", help="comma-separated subset (default: all)")
    parser.add_argument("--variants", default="", help="comma-separated subset (default: all)")
    parser.add_argument("--out", default="", help="write the summary JSON here")
    args = parser.parse_args()

    ops = [o for o in args.ops.split(",") if o] or list(available_targets())
    want_variants = [v for v in args.variants.split(",") if v]
    pairs = [
        (op, variant)
        for op in ops
        for variant in target_variants(op)
        if not want_variants or variant in want_variants
    ]

    summary: dict[str, dict[str, object]] = {}
    ok = True
    for op, variant in pairs:
        result = score_candidate_source(
            target_source(op, variant),
            op=op,
            device=args.device,
            timeout_s=TIMEOUT_S.get(op, 1400.0),
            measure_speedup=True,
        )
        passed = bool(result["contracts_passed"])
        bench = result.get("bench", {}) or {}
        speedup = result.get("speedup")
        failed_gates = {
            g: r["reason"]
            for g, r in result["gates"].items()
            if not r["passed"] and g.rsplit("/", 1)[-1] not in DEFAULT_EXCLUDE_GATES
        }
        # Expectation gate per variant: triton must clear parity (continuous
        # reward >= ~0.95), eager must hold the slow floor (== 0.5).
        if variant == "triton":
            ok &= passed and float(result["reward"]) >= 0.95
        elif variant == "eager":
            ok &= passed and abs(float(result["reward"]) - 0.5) < 1e-6

        summary[f"{op}[{variant}]"] = {
            "contracts_passed": passed,
            "reward": result["reward"],
            "speedup": speedup,
            "t_candidate_ms": bench.get("t_candidate_ms"),
            "t_baseline_ms": bench.get("t_baseline_ms"),
            "correct_at_bench": bench.get("correct_at_bench"),
            "bug_routing": result.get("bug_routing"),
            "status": result["status"],
            "views": f"{result['views_passed']}/{result['views_total']}",
            "error": result["error"],
            "failed_gates": failed_gates,
        }
        print(
            f"{op}[{variant}]: passed={passed} reward={float(result['reward']):.4f} "
            f"speedup={speedup} correct_at_bench={bench.get('correct_at_bench')} "
            f"bug_routing={result.get('bug_routing')} views={result['views_passed']}/{result['views_total']} "
            f"status={result['status']}"
        )
        if result["error"]:
            print(f"  ERROR: {result['error']}")
        for g, reason in failed_gates.items():
            print(f"  FAIL {g}: {reason}")
        sys.stdout.flush()

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    print(f"SMOKE_OK={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
