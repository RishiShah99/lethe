"""Measure kernel-vs-oracle scale_rel per C6 view at CMP-03's gate shapes.

CMP-03 drove the grad_C view 1.7-1.9e-5 of output scale over its 1e-5
default unit atol on B200 — grad_C contracts dys against the forward
chain state h, the largest compounding intermediate, so its cross-impl
reorder noise rides h's accumulated divergence. This prints the worst
scale_rel per view over the gate's five shapes x 5 draws — the numbers
behind any per-view CMP-03 override. Box usage:
fleet run "bash scratch/detach.sh uv run python scratch/c6_cmp03_probe.py"
"""

from __future__ import annotations

import torch

from flash_mamba_rl.kernels.ops import fused_block_backward
from flash_mamba_rl.verifier.op_harness import (
    FUSED_BWD_GRAD_FIELDS,
    fused_bwd_candidate_adapter,
    fused_bwd_reference_adapter,
)

SHAPES = [(1, 16, 8), (2, 32, 16), (4, 64, 32), (8, 128, 16), (1, 256, 64)]


def main() -> None:
    dev = torch.device("cuda")
    print(f"{'view':18s} " + " ".join(f"{s!s:>14}" for s in SHAPES) + "      worst")
    for field in FUSED_BWD_GRAD_FIELDS:
        cand = fused_bwd_candidate_adapter(fused_block_backward, field)
        ref = fused_bwd_reference_adapter(field)
        per_shape = []
        for shape in SHAPES:
            worst = 0.0
            for seed in range(5):
                torch.manual_seed(seed)
                dy = torch.randn(shape, device=dev)
                got = cand(dy)
                want = ref(dy)
                max_err = (got - want).abs().max().item()
                scale = want.abs().max().clamp(min=1.0).item()
                worst = max(worst, max_err / scale)
            per_shape.append(worst)
        print(
            f"{field:18s} " + " ".join(f"{v:14.3e}" for v in per_shape) + f"  {max(per_shape):9.3e}"
        )


if __name__ == "__main__":
    main()
