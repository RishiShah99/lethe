"""Desk de-risk for the channel-wise box kernels — orchestration logic on CPU, no box.

The tcgen05 GEMMs are box-only, but the *host orchestration* around them (operand layout:
cat order, transposes, key-axis decay folding, padding, reductions) is where the scalar
bring-up burned box cycles. This runs ``run_k1_incB`` / ``run_k2`` (channel-wise) on CPU with
the proven (128,64,128) GEMM swapped for a plain fp32 matmul, and grades the assembled
backward against the token-serial GDN-2 oracle. If the orchestration is wrong, it shows here
for free; the box then only has to confirm the tcgen05 GEMM numerics.

Also asserts the in-file ``*_cw_ref`` specs equal the assembly's closed forms bit-for-bit.

    PYTHONPATH="src;." uv run --no-sync python scratch/gdn2_cw_orchestration_check.py
"""

from __future__ import annotations

import torch

import lethe.kernels.cute.gdn2_bwd_dhu as k1base
import lethe.kernels.cute.gdn2_bwd_dhu_cw as k1cw
import lethe.kernels.cute.gdn2_bwd_wy as k2mod
import lethe.kernels.cute.gdn2_bwd_wy_cw as k2cw
from lethe.kernels.cute.gdn2_assemble import (
    assemble_gdn2_backward_channelwise,
    k1_reverse_state_cw_ref,
    k2_wy_vjp_cw_ref,
)
from lethe.kernels.references.gdn_backward import reference_gdn2_backward

SHAPES = [
    (1, 64, 1, 128, 128, 64),  # crown tile: d_k=d_v=128, C=64, NT=1
    (2, 128, 2, 128, 64, 64),  # NT=2, d_v=64 (the scalar tile width)
    (1, 192, 1, 128, 128, 64),  # NT=3
]


def _inputs(shape, seed, dtype=torch.float64):
    b, t, h, d_k, d_v, _ = shape
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(b, t, h, d_k, generator=gen, dtype=dtype)
    k = torch.randn(b, t, h, d_k, generator=gen, dtype=dtype)
    v = torch.randn(b, t, h, d_v, generator=gen, dtype=dtype)
    g = -(torch.rand(b, t, h, d_k, generator=gen, dtype=dtype) * 0.1 + 0.01)
    bg = torch.rand(b, t, h, d_k, generator=gen, dtype=dtype) * 0.8 + 0.1
    wg = torch.rand(b, t, h, d_v, generator=gen, dtype=dtype) * 0.8 + 0.1
    do = torch.randn(b, t, h, d_v, generator=gen, dtype=dtype)
    return q, k, v, g, bg, wg, do


def _fake_mm_tc(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Stand-in for the tcgen05 (128,64,128) GEMM: plain fp32 matmul, same contract."""
    return x.to(torch.float32) @ y.to(torch.float32)


def _fake_bmm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Stand-in for the batched (128,64,128) GEMM (lever B): fp32 ``x[Z,M,K] @ y[Z,K,N]``."""
    return x.to(torch.float32) @ y.to(torch.float32)


def main() -> None:
    # 1. The in-file specs equal the assembly's closed forms (bit-for-bit on fp64).
    q, k, v, g, bg, wg, do = _inputs(SHAPES[0], seed=1)
    s = SHAPES[0][3] ** -0.5
    from lethe.kernels.references.gdn2_chunkwise_cw import build_microgate_bundles_cw

    def _l2(x: torch.Tensor) -> torch.Tensor:
        return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)

    bundles = build_microgate_bundles_cw(_l2(q), _l2(k), v, g, bg, wg, do, chunk_len=64, scale=s)
    i1 = bundles["k1"].inputs
    a = k1cw.run_k1_cw_ref(
        i1["q"], i1["k"], i1["wy"], i1["g2"], i1["g_last"], i1["do"], i1["dv_local"], i1["dht"]
    )
    e = k1_reverse_state_cw_ref(
        i1["q"], i1["k"], i1["wy"], i1["g2"], i1["g_last"], i1["do"], i1["dv_local"], i1["dht"]
    )
    assert all(torch.equal(x, y) for x, y in zip(a, e, strict=True)), "run_k1_cw_ref != k1 closed form"
    i2 = bundles["k2"].inputs
    a2 = k2cw.run_k2_cw_ref(i2["k"], i2["v"], i2["b"], i2["w"], i2["g2"], i2["T"], i2["dwy"], i2["du"])
    e2 = k2_wy_vjp_cw_ref(i2["k"], i2["v"], i2["b"], i2["w"], i2["g2"], i2["T"], i2["dwy"], i2["du"])
    assert all(torch.equal(x, y) for x, y in zip(a2, e2, strict=True)), "run_k2_cw_ref != k2 closed form"
    print("specs == assembly closed forms: OK")

    # 2. Host orchestration on CPU (GEMM -> fp32 matmul), assembled grads vs oracle.
    orig_avail_k1 = k1cw.is_available
    orig_avail_k2 = k2cw.is_available
    orig_mm = k2mod._mm_tc
    orig_bmm = k1base._bmm_tc  # the batched GEMM the default cw path now routes through (lever B)
    orig_sync = torch.cuda.synchronize
    k1cw.is_available = lambda: True
    k2cw.is_available = lambda: True
    k2mod._mm_tc = _fake_mm_tc
    k1base._bmm_tc = _fake_bmm
    torch.cuda.synchronize = lambda *a, **k: None
    try:
        worst = 0.0
        for shape in SHAPES:
            q, k, v, g, bg, wg, do = _inputs(shape, seed=shape[1])
            s = shape[3] ** -0.5
            grads = assemble_gdn2_backward_channelwise(
                q, k, v, g, bg, wg, do, scale=s, use_qk_l2norm=True,
                k1_fn=k1cw.run_k1_incB, k2_fn=k2cw.run_k2,
            )
            orc = reference_gdn2_backward(q, k, v, g, bg, wg, do, scale=s, use_qk_l2norm=True)
            for got, ref in (
                (grads.dq, orc.grad_q), (grads.dk, orc.grad_k), (grads.dv, orc.grad_v),
                (grads.dg, orc.grad_g), (grads.db, orc.grad_b), (grads.dw, orc.grad_w),
            ):
                rel = ((got - ref).abs().max() / ref.abs().max().clamp_min(1e-12)).item()
                worst = max(worst, rel)
            print(f"  shape={shape} worst_rel so far={worst:.2e}")
        ok = worst < 1e-6
        print(f"\nhost-orchestration (fake GEMM) vs oracle: worst_scale_rel={worst:.2e} GO={ok}")
        assert ok, "orchestration logic diverges from the oracle"
    finally:
        k1cw.is_available = orig_avail_k1
        k2cw.is_available = orig_avail_k2
        k2mod._mm_tc = orig_mm
        k1base._bmm_tc = orig_bmm
        torch.cuda.synchronize = orig_sync


if __name__ == "__main__":
    main()
