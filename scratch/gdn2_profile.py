"""#52 follow-up — where does the cw backward's DEVICE time go at large L?

The graph bench showed graphing collapses host overhead (3x at L512) but only 1.3x at
L2048 — at large L the cost is device compute, not dispatch. This profiles ONE eager cw
backward at a large shape and prints the CUDA-time breakdown, to prioritize the device-side
levers: #51 (skip forward re-stage), #50 (cut per-chunk torch glue / einsums), #49 (bigger
GEMM tiles). Also times the forward re-stage alone (the #51 candidate) vs the full backward.

  PYTHONPATH=src:. ~/cuteenv/bin/python scratch/gdn2_profile.py
"""

from __future__ import annotations

import torch

D_K = D_V = 128


def _inputs(b, t, h, dtype, dev, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    mk = lambda d: torch.randn(b, t, h, d, generator=gen)  # noqa: E731
    q, k, v = mk(D_K), mk(D_K), mk(D_V)
    g = -torch.rand(b, t, h, D_K, generator=gen) * 0.1
    bg = torch.rand(b, t, h, D_K, generator=gen).sigmoid()
    wg = torch.rand(b, t, h, D_V, generator=gen).sigmoid()
    do = torch.randn(b, t, h, D_V, generator=gen)
    return tuple(x.to(device=dev, dtype=dtype).contiguous() for x in (q, k, v, g, bg, wg, do))


def main() -> None:
    # No argparse: cutlass-dsl runs its own import-time arg parser over sys.argv, so a custom
    # CLI flag collides with it. Hardcode the large shape (the device-bound regime to profile).
    bb, ll, hh = 2, 2048, 8
    if not torch.cuda.is_available():
        print("no CUDA (desk)")
        return

    from torch.profiler import ProfilerActivity, profile

    from flash_mamba_rl.kernels.cute.gdn2_assemble import (
        assembled_channelwise_gdn2_backward,
        chunkwise_forward_cw,
    )
    from flash_mamba_rl.kernels.cute.gdn2_backward import _load_box_kernels_cw

    k1_cw, k2_cw = _load_box_kernels_cw()
    dev = torch.device("cuda")
    q, k, v, g, bg, wg, do = _inputs(bb, ll, hh, torch.bfloat16, dev, seed=7)

    def full():
        return assembled_channelwise_gdn2_backward(
            q, k, v, g, bg, wg, do, use_qk_l2norm=True, k1_fn=k1_cw, k2_fn=k2_cw
        )

    full()  # warm / JIT
    torch.cuda.synchronize()

    # Forward re-stage alone (the #51 lever target): fraction of the backward this is.
    qf, kf, vf, gf, bgf, wgf = (x.float() for x in (q, k, v, g, bg, wg))
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    ev[0].record()
    for _ in range(5):
        chunkwise_forward_cw(qf, kf, vf, gf, bgf, wgf, scale=None, use_qk_l2norm=True)
    ev[1].record()
    torch.cuda.synchronize()
    fwd_ms = ev[0].elapsed_time(ev[1]) / 5

    ev[2].record()
    for _ in range(5):
        full()
    ev[3].record()
    torch.cuda.synchronize()
    full_ms = ev[2].elapsed_time(ev[3]) / 5
    print(
        f"shape=(b{bb},L{ll},H{hh})  full_bwd={full_ms:.2f}ms  "
        f"forward_restage={fwd_ms:.2f}ms ({100 * fwd_ms / full_ms:.0f}% of bwd)"
    )

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        full()
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))


if __name__ == "__main__":
    main()
