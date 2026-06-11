"""Time the rope kernel at num_warps 2 vs 4 across block regimes.

The B200 bench measured nw=2 fastest at the flagship shape
(B8xL2048xH32xP64xN128: 6.48 ms vs 7.60 ms at the nw=4 default) — this
adds the small-block datapoints the launcher heuristic needs before its
default changes. Box usage:
fleet run "bash scratch/detach.sh uv run python scratch/c4_warp_probe.py"
"""

from __future__ import annotations

import torch

from flash_mamba_rl.bench.c2_backward_selective_scan import _time
from flash_mamba_rl.kernels.ops import _triton_complex_rope

SHAPES = [
    (2, 256, 4, 16, 32, 16),  # oracle shape: block 16x32=512, nw4 branch
    (2, 1024, 8, 4, 16, 6),  # gate-like: block 4x16=64, nw2 branch
    (4, 2048, 16, 64, 128, 64),  # mid
    (8, 2048, 32, 64, 128, 64),  # flagship (bench cross-check)
]


def main() -> None:
    dev = torch.device("cuda")
    for b, length, h, p, n, s in SHAPES:
        torch.manual_seed(0)
        x = torch.randn(b, length, h, p, device=dev)
        bb = torch.randn(b, length, h, n, device=dev)
        cc = torch.randn(b, length, h, n, device=dev)
        dt = torch.rand(b, length, h, device=dev) * 0.1 + 1e-3
        a = -torch.rand(h, device=dev)
        angle = torch.randn(b, length, h, s, device=dev)
        args = (x, bb, cc, dt, a, angle)
        row = []
        for warps in (2, 4):

            def run(w: int = warps, inputs: tuple[torch.Tensor, ...] = args) -> torch.Tensor:
                return _triton_complex_rope.launch_complex_scan_rope(*inputs, num_warps=w)

            t = _time(run, warmup=5, trials=30)
            row.append(f"nw={warps}: {t['median_ms']:8.3f} ms")
        print(f"B{b}xL{length}xH{h}xP{p}xN{n}xS{s}: " + "  ".join(row))


if __name__ == "__main__":
    main()
