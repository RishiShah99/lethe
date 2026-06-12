"""Score every SFT warm-start target through the CUDA verifier.

The hard gate before any SFT step consumes a target: each must come back
contracts_passed at reward >= 0.5 from the exact scoring path GRPO uses
(score_candidate_source, sandboxed, device=cuda). A target that does not
pass teaches nothing.

Usage (box):
    uv run python scratch/sft_validate.py [--device cuda] [--ops op1,op2]
    uv run python scratch/sft_validate.py --out sft_validation.json
"""

from __future__ import annotations

import argparse
import json
import sys

from flash_mamba_rl.rl.sft_targets import available_targets, target_source
from flash_mamba_rl.verifier.candidate_scoring import (
    DEFAULT_EXCLUDE_GATES,
    score_candidate_source,
)

# Mirrors phase_e_run.SCORE_TIMEOUT_S, doubled: the eager target pays the
# reference's own autograd cost a second time on the candidate side.
TIMEOUT_S: dict[str, float] = {
    "forward_chunked_scan": 840.0,
    "complex_scan_rope": 840.0,
    "fused_block_forward": 1200.0,
    "backward_selective_scan": 1800.0,
    "mimo_backward": 2000.0,
    "fused_block_backward": 2800.0,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ops", default="", help="comma-separated subset (default: all)")
    parser.add_argument("--out", default="", help="write the summary JSON here")
    args = parser.parse_args()

    ops = [o for o in args.ops.split(",") if o] or list(available_targets())
    summary: dict[str, dict[str, object]] = {}
    all_passed = True
    for op in ops:
        result = score_candidate_source(
            target_source(op),
            op=op,
            device=args.device,
            timeout_s=TIMEOUT_S.get(op, 1200.0),
        )
        passed = bool(result["contracts_passed"])
        all_passed &= passed
        # Default-excluded gates (CMP-02) fail by construction on backward
        # views and never charge the verdict — keep them out of the log.
        failed_gates = {
            g: r["reason"]
            for g, r in result["gates"].items()
            if not r["passed"] and g.rsplit("/", 1)[-1] not in DEFAULT_EXCLUDE_GATES
        }
        summary[op] = {
            "contracts_passed": passed,
            "reward": result["reward"],
            "status": result["status"],
            "views": f"{result['views_passed']}/{result['views_total']}",
            "error": result["error"],
            "failed_gates": failed_gates,
        }
        print(f"{op}: passed={passed} reward={result['reward']} status={result['status']}")
        if failed_gates:
            for g, reason in failed_gates.items():
                print(f"  FAIL {g}: {reason}")
        sys.stdout.flush()

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    print(f"ALL_PASSED={all_passed}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
