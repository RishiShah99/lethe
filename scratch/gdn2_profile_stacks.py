"""#50a — pin the cw backward's DEVICE cost to its Python call sites (with_stack profiler).

#52's profile (gdn2_profile.py) showed the eager cw backward's device time is ~53% cuBLAS
matmuls (a 27% ``gemv`` cluster + 22% ``gemmk1`` + 4% simt-sgemm), ~38% elementwise glue, and
only 2.5% our tcgen05 kernels. To cut it we need to know WHICH Python line emits the gemv
cluster — the HANDOFF hypothesis is a skinny fp32 matmul in one of the two per-chunk loops
(``chunkwise_forward_cw`` re-stage or ``_stage_b_vjp_cw``). This does two passes:

  1. a clean per-stage CUDA-event breakdown (re-stage / stage-B VJP / dv_local / K#1 / K#2 /
     L2-VJP) so each stage's device cost is unambiguous, no stack heuristics;
  2. a ``with_stack=True`` torch.profiler pass that groups CUDA kernels by Python call site
     (``group_by_stack_n``) and exports a foldable stack file, pinning the gemv cluster.

Hardcoded shape (no argparse: cutlass-dsl parses sys.argv at import). Box-only.

  PYTHONPATH=src:. ~/cuteenv/bin/python scratch/gdn2_profile_stacks.py
"""

from __future__ import annotations

import torch

D_K = D_V = 128
OUT_STACKS = "results/gdn2_stacks_cuda.txt"
OUT_JSON = "results/gdn2_profile_stacks.json"


def _inputs(b, t, h, dev, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    mk = lambda d: torch.randn(b, t, h, d, generator=gen)  # noqa: E731
    q, k, v = mk(D_K), mk(D_K), mk(D_V)
    g = -torch.rand(b, t, h, D_K, generator=gen) * 0.1
    bg = torch.rand(b, t, h, D_K, generator=gen).sigmoid()
    wg = torch.rand(b, t, h, D_V, generator=gen).sigmoid()
    do = torch.randn(b, t, h, D_V, generator=gen)
    # The assembly upcasts half inputs to fp32 internally; profile that fp32 path directly.
    return tuple(x.to(device=dev, dtype=torch.float32).contiguous() for x in (q, k, v, g, bg, wg, do))


def _time(fn, n=5):
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    fn()
    torch.cuda.synchronize()
    ev[0].record()
    for _ in range(n):
        fn()
    ev[1].record()
    torch.cuda.synchronize()
    return ev[0].elapsed_time(ev[1]) / n


def main() -> None:
    import json

    bb, ll, hh = 2, 2048, 8
    if not torch.cuda.is_available():
        print("no CUDA (desk)")
        return

    from torch.profiler import ProfilerActivity, profile

    from lethe.kernels.cute.gdn2_assemble import (
        _stage_b_vjp_cw,
        _to_chunks,
        assembled_channelwise_gdn2_backward,
        chunkwise_forward_cw,
        pick_chunk_len,
    )
    from lethe.kernels.cute.gdn2_backward import _load_box_kernels_cw

    k1_cw, k2_cw = _load_box_kernels_cw()
    dev = torch.device("cuda")
    q, k, v, g, bg, wg, do = _inputs(bb, ll, hh, dev, seed=7)
    cl = pick_chunk_len(ll)

    def full():
        gr = assembled_channelwise_gdn2_backward(
            q, k, v, g, bg, wg, do, use_qk_l2norm=True, k1_fn=k1_cw, k2_fn=k2_cw
        )
        return (gr.grad_q, gr.grad_k, gr.grad_v, gr.grad_g, gr.grad_b, gr.grad_w)

    # ---- Per-stage CUDA-event breakdown (the assembly's internal fp32 sequence) ----
    full_ms = _time(full)

    fwd = chunkwise_forward_cw(q, k, v, g, bg, wg, chunk_len=cl, use_qk_l2norm=True)
    restage_ms = _time(
        lambda: chunkwise_forward_cw(q, k, v, g, bg, wg, chunk_len=cl, use_qk_l2norm=True)
    )
    stage_b_ms = _time(lambda: _stage_b_vjp_cw(fwd, do, create_graph=False))

    do_c = _to_chunks(do, cl)
    dvloc_ms = _time(lambda: fwd.A_qk.transpose(-1, -2) @ do_c)

    _dqb, _dkb, _dgb, dwy, _du, _dh0 = _stage_b_vjp_cw(fwd, do, create_graph=False)
    dv_local = fwd.A_qk.transpose(-1, -2) @ do_c
    dht = torch.zeros_like(fwd.h_list[0])
    fq, fk, fwy, fg2, fgl, fb, fwg, fv, fT = (
        fwd.q.detach(), fwd.k.detach(), fwd.wy.detach(), fwd.g2.detach(), fwd.g_last.detach(),
        fwd.b.detach(), fwd.w_gate.detach(), fwd.v.detach(), fwd.T.detach(),
    )
    do_cd, dvl_d, dht_d = do_c.detach(), dv_local.detach(), dht.detach()

    def run_k1():
        return k1_cw(fq, fk, fwy, fg2, fgl, do_cd, dvl_d, dht_d)

    _dh, dv2, _dh0k = run_k1()
    dv2_d = dv2.detach()
    dwy_d = dwy.detach()
    k1_ms = _time(run_k1)
    k2_ms = _time(lambda: k2_cw(fk, fv, fb, fwg, fg2, fT, dwy_d, dv2_d))

    s = D_K**-0.5
    dq_n = torch.randn(bb, ll, hh, D_K, device=dev)
    dk_n = torch.randn(bb, ll, hh, D_K, device=dev)

    def l2_vjp():
        q_lf = q.detach().clone().requires_grad_(True)
        k_lf = k.detach().clone().requires_grad_(True)
        q_sn = (q_lf / torch.sqrt((q_lf * q_lf).sum(-1, keepdim=True) + 1e-6)) * s
        k_nn = k_lf / torch.sqrt((k_lf * k_lf).sum(-1, keepdim=True) + 1e-6)
        return torch.autograd.grad((q_sn, k_nn), (q_lf, k_lf), (dq_n, dk_n))

    l2_ms = _time(l2_vjp)

    stages = {
        "forward_restage": restage_ms,
        "stage_b_vjp": stage_b_ms,
        "dv_local": dvloc_ms,
        "k1_reverse_scan": k1_ms,
        "k2_wy_vjp": k2_ms,
        "l2norm_vjp": l2_ms,
    }
    summed = sum(stages.values())
    print(f"\nshape=(b{bb},L{ll},H{hh})  full_bwd={full_ms:.2f}ms  sum_of_stages={summed:.2f}ms")
    for name, ms in sorted(stages.items(), key=lambda kv: -kv[1]):
        print(f"  {name:18s} {ms:8.2f}ms  {100 * ms / full_ms:5.1f}% of full  {100 * ms / summed:5.1f}% of sum")

    # ---- with_stack pass: pin each CUDA kernel cluster to its Python call site ----
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=True,
        record_shapes=True,
    ) as prof:
        full()
        torch.cuda.synchronize()

    print("\n==== TOP CUDA kernels by self time (flat) ====")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
    print("\n==== TOP CUDA kernels grouped by Python call site (group_by_stack_n=14) ====")
    print(
        prof.key_averages(group_by_stack_n=14).table(
            sort_by="self_cuda_time_total", row_limit=30
        )
    )
    try:
        prof.export_stacks(OUT_STACKS, "self_cuda_time_total")
        print(f"\nwrote foldable stacks -> {OUT_STACKS}")
    except Exception as e:
        print(f"export_stacks failed: {e!r}")

    with open(OUT_JSON, "w") as f:
        json.dump(
            {"shape": [bb, ll, hh], "full_ms": full_ms, "stages_ms": stages}, f, indent=2
        )
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
