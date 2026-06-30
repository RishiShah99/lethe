"""#48 isolation — capture the cw backward for ONE shape/mode in a FRESH process.

The combined gate failed BOTH l2norm modes, but a failed capture leaves the stream in a
capturing/errored state that poisons every later capture in the same process. This runs
exactly ONE (l2norm, shape) per process so the verdict is clean, prints the REAL traceback
(no swallow), and is meant to run under CUDA_LAUNCH_BLOCKING=1 so the first failing op is
reported at its true site. The hypothesis: ``use_qk_l2norm=True`` (the torch.autograd.grad
L2-norm VJP) is the blocker; ``use_qk_l2norm=False`` should capture clean.

  CUDA_LAUNCH_BLOCKING=1 PYTHONPATH=src:. ~/cuteenv/bin/python scratch/graph_cw_probe.py --l2 0
  CUDA_LAUNCH_BLOCKING=1 PYTHONPATH=src:. ~/cuteenv/bin/python scratch/graph_cw_probe.py --l2 1
"""

from __future__ import annotations

import argparse
import traceback

import torch

D_K = D_V = 128
GRID = [(1, 64, 1), (1, 128, 2), (2, 128, 1), (1, 256, 1)]


def _inputs(b, t, h, dtype, dev, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(b, t, h, D_K, generator=gen)
    k = torch.randn(b, t, h, D_K, generator=gen)
    v = torch.randn(b, t, h, D_V, generator=gen)
    g = -torch.rand(b, t, h, D_K, generator=gen) * 0.1
    bg = torch.rand(b, t, h, D_K, generator=gen).sigmoid()
    wg = torch.rand(b, t, h, D_V, generator=gen).sigmoid()
    do = torch.randn(b, t, h, D_V, generator=gen)
    return tuple(x.to(device=dev, dtype=dtype).contiguous() for x in (q, k, v, g, bg, wg, do))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--l2", type=int, default=0)
    ap.add_argument("--shape-idx", type=int, default=0)
    args = ap.parse_args()
    use_l2 = bool(args.l2)

    if not torch.cuda.is_available():
        print("no CUDA (desk)")
        return

    from scratch.gdn2_graph import GraphedBackward

    from flash_mamba_rl.kernels.cute.gdn2_assemble import assembled_channelwise_gdn2_backward
    from flash_mamba_rl.kernels.cute.gdn2_backward import _load_box_kernels_cw

    # Bypass native_gdn2_backward's dispatch: its _is_scalar_reducible does torch.allclose
    # (a device->host sync) which is illegal during capture. We KNOW the cw regime here, so
    # call the cw assembly directly with the box kernels (the dispatch decision is made once,
    # outside the graph).
    k1_cw, k2_cw = _load_box_kernels_cw()

    dev = torch.device("cuda")
    b, t, h = GRID[args.shape_idx]
    inp = _inputs(b, t, h, torch.bfloat16, dev, seed=b * 100 + t)

    def adapter(q, k, v, g, bb, w, do):
        gr = assembled_channelwise_gdn2_backward(
            q, k, v, g, bb, w, do, use_qk_l2norm=use_l2, k1_fn=k1_cw, k2_fn=k2_cw
        )
        return (gr.grad_q, gr.grad_k, gr.grad_v, gr.grad_g, gr.grad_b, gr.grad_w)

    print(f"=== l2norm={use_l2} shape=({b},{t},{h}) ===")
    eager = adapter(*inp)
    torch.cuda.synchronize()
    print("EAGER_OK n_out=", len(eager))

    graphed = GraphedBackward(adapter)
    try:
        g1 = graphed(*inp)
        torch.cuda.synchronize()
        rel = max(
            (
                (g1[i].float() - eager[i].float()).abs().max()
                / eager[i].float().abs().max().clamp_min(1e-12)
            ).item()
            for i in range(len(eager))
        )
        print(f"CAPTURE_OK rel_vs_eager={rel:.3e}")
    except Exception:
        traceback.print_exc()
        print("CAPTURE_FAIL")


if __name__ == "__main__":
    main()
