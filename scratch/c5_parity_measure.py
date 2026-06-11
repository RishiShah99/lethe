"""Measure the kernel-vs-reference fp32 parity at the C5 GPU-test shapes.

Prints max_err and scale_rel per shape over several seeds — the numbers
behind TestC5TritonParity's 1e-4 scale-rel bound (the C1-C4 pattern of
pinning the bound from B200 measurement). Box usage:
fleet run "bash scratch/detach.sh uv run python scratch/c5_parity_measure.py"
"""

from __future__ import annotations

import math

import torch

from flash_mamba_rl.kernels.ops import fused_block_forward
from flash_mamba_rl.kernels.references import reference_fused_block_forward

# (batch, l_out, d_model, n_state, conv_k, chunk_size)
SHAPES = [
    (1, 8, 4, 8, 4, 8),
    (2, 64, 96, 16, 4, 8),
    (3, 128, 100, 10, 3, 8),
    (2, 256, 512, 32, 4, 8),
    (2, 120, 300, 16, 4, 8),
    (1, 13, 36, 8, 4, 13),
    (2, 64, 48, 16, 1, 8),
    (8, 2048, 4096, 128, 4, 64),  # training shape
]


def main() -> None:
    dev = torch.device("cuda")
    worst = 0.0
    for b, l_out, d, n, k, chunk in SHAPES:
        shape_worst = 0.0
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
            args = (x, conv_w, conv_b, delta, a, b_proj, c_proj, d_skip, norm_w)
            got = fused_block_forward(*args, conv_kernel_size=k, chunk_size=chunk)
            want = reference_fused_block_forward(*args, conv_kernel_size=k, chunk_size=chunk)
            max_err = (got - want).abs().max().item()
            scale = want.abs().max().clamp(min=1.0).item()
            shape_worst = max(shape_worst, max_err / scale)
        worst = max(worst, shape_worst)
        print(f"shape=({b},{l_out},{d},{n},K{k}) worst scale_rel={shape_worst:9.3e}")
    print(f"\noverall worst scale_rel: {worst:9.3e}")


if __name__ == "__main__":
    main()
