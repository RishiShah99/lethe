"""Measure the kernel-vs-reference fp32 parity at the C4 GPU-test shapes.

Prints max_err and scale_rel per shape over several seeds — the numbers
behind tightening TestC4TritonParity's provisional 1e-3 scale-rel bound
from B200 measurement (the C1-C3 pattern). Box usage:
fleet run "bash scratch/detach.sh uv run python scratch/c4_parity_measure.py"
"""

from __future__ import annotations

import torch

from lethe.kernels.ops import complex_scan_rope
from lethe.kernels.references.complex_scan_rope import reference_complex_scan_rope

SHAPES = [
    (1, 8, 2, 4, 8, 3),
    (2, 64, 3, 24, 16, 8),
    (3, 121, 4, 100, 10, 4),
    (2, 256, 8, 64, 128, 64),
    (2, 64, 2, 80, 16, 6),
    (1, 1, 2, 4, 8, 2),
    (8, 2048, 32, 64, 128, 64),
]


def main() -> None:
    dev = torch.device("cuda")
    worst = 0.0
    for b, seq, h, p, n, s in SHAPES:
        shape_worst = 0.0
        for seed in range(5):
            torch.manual_seed(seed)
            x = torch.randn(b, seq, h, p, device=dev)
            bb = torch.randn(b, seq, h, n, device=dev)
            cc = torch.randn(b, seq, h, n, device=dev)
            dt = torch.rand(b, seq, h, device=dev) * 0.1 + 1e-3
            a = -torch.rand(h, device=dev)
            angle = torch.randn(b, seq, h, s, device=dev)
            got = complex_scan_rope(x, bb, cc, dt, a, angle)
            want = reference_complex_scan_rope(x, bb, cc, dt, a, angle)
            max_err = (got - want).abs().max().item()
            scale = want.abs().max().clamp(min=1.0).item()
            shape_worst = max(shape_worst, max_err / scale)
        worst = max(worst, shape_worst)
        print(f"shape=({b},{seq},{h},{p},{n},{s}) worst scale_rel={shape_worst:9.3e}")
    print(f"\noverall worst scale_rel: {worst:9.3e}")


if __name__ == "__main__":
    main()
