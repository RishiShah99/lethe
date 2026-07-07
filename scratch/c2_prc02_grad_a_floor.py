"""Diagnose the grad_A PRC-02 failure on B200: input-rounding vs cancellation.

For the backward op's grad_A view at the PRC-02 shape (saturation-free),
prints where the candidate-vs-oracle error actually lives:

- err at the global-max element and |ref| there,
- max err restricted to |ref| <= 1 (cancellation elements),
- max relative err restricted to |ref| > 1 (scale-carried elements),
- the same stats for the eager fallback (no Triton reorder noise) and the
  fp16-carry cheat (the signal PRC-02 must keep rejecting).

The tolerance fix differs by diagnosis: scale-carried error wants rtol,
cancellation error wants output-scale-aware atol (the C1 CMP-01 lesson).

Run on the box: uv run python scratch/c2_prc02_grad_a_floor.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_op_harness import _bwd_via_autograd, _fp16_accumulator_scan

from lethe.kernels.ops import backward_selective_scan
from lethe.verifier.op_harness import (
    SCAN_BWD_GATE_OVERRIDES,
    bwd_scan_candidate_adapter,
    bwd_scan_reference_adapter,
)

SHAPE = SCAN_BWD_GATE_OVERRIDES["grad_A"]["gate_prc_02_mixed_precision_accumulation"]["shape"]


def stats(bwd_fn, label: str, device: str) -> None:
    candidate = bwd_scan_candidate_adapter(bwd_fn, "grad_A", saturate=False)
    reference = bwd_scan_reference_adapter("grad_A", saturate=False)

    torch.manual_seed(0)
    dy32 = torch.randn(SHAPE, device=device)
    ref = reference(dy32).float()
    cand = candidate(dy32.to(torch.float16)).float()

    diff = (cand - ref).abs()
    flat_idx = int(diff.argmax())
    ref_at_max = float(ref.flatten()[flat_idx])
    small = ref.abs() <= 1.0
    large = ~small
    max_err_small = float(diff[small].max()) if small.any() else float("nan")
    max_rel_large = float((diff[large] / ref[large].abs()).max()) if large.any() else float("nan")
    print(
        f"{label:<24} dev={device:<5} |ref|_inf={float(ref.abs().max()):>9.3f} "
        f"max_err={float(diff.max()):.4e} @|ref|={abs(ref_at_max):.3f}  "
        f"max_err(|ref|<=1)={max_err_small:.4e}  max_rel(|ref|>1)={max_rel_large:.4e}"
    )


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stats(backward_selective_scan, "ours (triton)" if device == "cuda" else "ours (eager)", device)
    if device == "cuda":
        stats(backward_selective_scan, "ours (eager/cpu)", "cpu")
    stats(_bwd_via_autograd(_fp16_accumulator_scan), "cheat (fp16 carry)", device)


if __name__ == "__main__":
    main()
