"""Measure kernel-vs-reference fp32 grad parity at the C6 GPU-test shapes.

Prints the worst per-gradient scale_rel per shape over several seeds — the
numbers behind TestC6TritonParity's 1e-4 scale-rel bound (the C1-C5
pattern of pinning the bound from B200 measurement). Box usage:
fleet run "bash scratch/detach.sh uv run python scratch/c6_parity_measure.py"
"""

from __future__ import annotations

import math

import torch

from flash_mamba_rl.kernels.ops import fused_block_backward
from flash_mamba_rl.kernels.references.fused_block_backward import (
    FusedBlockGrads,
    reference_fused_block_backward,
)

# (batch, l_out, d_model, n_state, conv_k, chunk_size)
SHAPES = [
    (1, 8, 4, 8, 4, 8),
    (2, 64, 96, 16, 4, 8),
    (3, 120, 100, 10, 3, 8),
    (2, 256, 512, 32, 4, 8),
    (2, 120, 300, 16, 4, 8),
    (1, 13, 36, 8, 4, 13),
    (2, 64, 48, 16, 1, 8),
    (8, 2048, 4096, 128, 4, 64),  # training shape
]


def main() -> None:
    dev = torch.device("cuda")
    worst = 0.0
    worst_field = ""
    for b, l_out, d, n, k, chunk in SHAPES:
        shape_worst = 0.0
        shape_field = ""
        for seed in range(5):
            torch.manual_seed(seed)
            x = torch.randn(b, l_out + k - 1, d, device=dev)
            conv_w = torch.randn(d, 1, k, device=dev) / math.sqrt(k)
            conv_b = 0.5 * torch.randn(d, device=dev)
            delta = torch.randn(b, l_out, d, device=dev)
            a = -torch.rand(d, n, device=dev)
            b_proj = torch.randn(b, l_out, n, device=dev)
            c_proj = torch.randn(b, l_out, n, device=dev)
            d_skip = torch.randn(d, device=dev)
            norm_w = 1.0 + 0.25 * torch.randn(d, device=dev)
            dy = torch.randn(b, l_out, d, device=dev)
            args = (x, conv_w, conv_b, delta, a, b_proj, c_proj, d_skip, norm_w, dy)
            got = fused_block_backward(*args, conv_kernel_size=k, chunk_size=chunk)
            want = reference_fused_block_backward(*args, conv_kernel_size=k, chunk_size=chunk)
            for field, g, r in zip(FusedBlockGrads._fields, got, want, strict=True):
                max_err = (g - r).abs().max().item()
                scale = r.abs().max().clamp(min=1.0).item()
                if max_err / scale > shape_worst:
                    shape_worst = max_err / scale
                    shape_field = field
        if shape_worst > worst:
            worst = shape_worst
            worst_field = shape_field
        print(
            f"shape=({b},{l_out},{d},{n},K{k}) worst scale_rel={shape_worst:9.3e} ({shape_field})"
        )
    print(f"\noverall worst scale_rel: {worst:9.3e} ({worst_field})")


if __name__ == "__main__":
    main()
