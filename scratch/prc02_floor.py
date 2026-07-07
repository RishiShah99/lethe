"""Measure PRC-02 floors for the scan op through the real harness.

Runs the actual gate (with the scan overrides) over the honest op and the
fp16-accumulator cheat, plus raw elementwise error stats, so the numbers
in SCAN_GATE_OVERRIDES comments stay tied to what the gate truly measures.
Run after any change to the aux distribution or the gate comparison.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_op_harness import _fp16_accumulator_scan

from lethe.kernels.ops import forward_chunked_scan
from lethe.verifier.contracts import gate_prc_02_mixed_precision_accumulation
from lethe.verifier.op_harness import (
    SCAN_GATE_OVERRIDES,
    scan_candidate_adapter,
    scan_reference_adapter,
)


def floor_stats(scan_fn, label: str) -> None:
    overrides = SCAN_GATE_OVERRIDES["gate_prc_02_mixed_precision_accumulation"]
    shape = overrides["shape"]
    rtol = overrides.get("rtol", 0.0)

    # saturate=False mirrors verify_scan_op's PRC-02 re-run: the gate
    # measures accumulator precision in the near-integrator regime only.
    candidate = scan_candidate_adapter(scan_fn, saturate=False)
    reference = scan_reference_adapter(saturate=False)

    result = gate_prc_02_mixed_precision_accumulation(
        candidate, reference, shape=shape, rtol=rtol
    )

    torch.manual_seed(0)
    t32 = torch.randn(shape)
    ref = reference(t32).float()
    cand = candidate(t32.to(torch.float16)).float()
    diff = (cand - ref).abs()
    margin = diff / (2e-2 + rtol * ref.abs())
    print(
        f"{label:<22} gate_passed={result.passed!s:<6} max_err={diff.max():.4e} "
        f"worst_margin={margin.max():.3f} (>1 fails)  reason={result.reason}"
    )


def main() -> None:
    floor_stats(forward_chunked_scan, "honest (fp32 acc)")
    floor_stats(_fp16_accumulator_scan, "cheat (fp16 acc)")


if __name__ == "__main__":
    main()
