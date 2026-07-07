"""Measure kernel-vs-oracle scale_rel per C6 view at CMP-03 and CMP-01 shapes.

CMP-03 drove the grad_C view 1.7-1.9e-5 of output scale over its 1e-5
default unit atol on B200 — grad_C contracts dys against the forward
chain state h, the largest compounding intermediate, so its cross-impl
reorder noise rides h's accumulated divergence. CMP-01 runs the same
comparison at its own shapes — its long_seq variation (4, 256, 32)
carries a longer worst chain (B*L^2/2 = 131072) than any CMP-03 shape,
so the same lesson must be measured there, not assumed (8 draws, the
gate's n_random). This prints the worst scale_rel per view per shape —
the numbers behind any per-view CMP-03/CMP-01 override. Box usage:
fleet run "bash scratch/detach.sh uv run python scratch/c6_cmp03_probe.py"
"""

from __future__ import annotations

import torch

from lethe.kernels.ops import fused_block_backward
from lethe.verifier.op_harness import (
    FUSED_BWD_GRAD_FIELDS,
    fused_bwd_candidate_adapter,
    fused_bwd_reference_adapter,
)

CMP03_SHAPES = [(1, 16, 8), (2, 32, 16), (4, 64, 32), (8, 128, 16), (1, 256, 64)]
CMP01_SHAPES = [(4, 64, 32), (4, 256, 32)]


def _table(label: str, shapes: list[tuple[int, int, int]], draws: int) -> None:
    dev = torch.device("cuda")
    print(f"--- {label}: {draws} draws/shape ---")
    print(f"{'view':18s} " + " ".join(f"{s!s:>14}" for s in shapes) + "      worst")
    for field in FUSED_BWD_GRAD_FIELDS:
        cand = fused_bwd_candidate_adapter(fused_block_backward, field)
        ref = fused_bwd_reference_adapter(field)
        per_shape = []
        for shape in shapes:
            worst = 0.0
            for seed in range(draws):
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


def main() -> None:
    _table("CMP-03 gate shapes", CMP03_SHAPES, 5)
    _table("CMP-01 base + long_seq", CMP01_SHAPES, 8)


if __name__ == "__main__":
    main()
