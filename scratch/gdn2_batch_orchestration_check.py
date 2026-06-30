"""Desk gate for lever B — the batched run_k1_incB / run_k2 (scalar + channel-wise) on CPU.

The tcgen05 batched GEMM is box-only, but the batching ORCHESTRATION (which tensors get
cat/transposed/decayed, the n_bh / n_idx batch packing) is pure torch. This swaps the batched
GEMM helper (``gdn2_bwd_dhu._bmm_tc``) and the serial helpers (``_mm_tc`` / ``_gemm_aa``) for
clean fp32 matmuls and checks the batched kernels against (a) the fp64 micro-gate references and
(b) the proven serial path. If the batching is wrong it shows here for free; the box then only has
to confirm the tcgen05 batched GEMM numerics (the same (128,64,128) config Phase 2/3 verified).

    PYTHONPATH="src;." uv run --no-sync python scratch/gdn2_batch_orchestration_check.py
"""

from __future__ import annotations

import scratch.gdn2_bwd_dhu as k1mod
import scratch.gdn2_bwd_dhu_cw as k1cw
import scratch.gdn2_bwd_wy as k2mod
import scratch.gdn2_bwd_wy_cw as k2cw
import torch

from flash_mamba_rl.kernels.references.gdn2_chunkwise import build_microgate_bundles
from flash_mamba_rl.kernels.references.gdn2_chunkwise_cw import build_microgate_bundles_cw

# (b, h, nt, c, d_k, d_v) — gentle decays so the fp32 GEMM stand-in stays near the fp64 ref.
SCALAR_SHAPES = [(1, 1, 1, 64, 128, 64), (2, 2, 4, 64, 128, 64), (1, 1, 8, 64, 128, 64)]
CW_SHAPES = [(1, 1, 1, 64, 128, 128), (2, 2, 2, 64, 128, 64), (1, 1, 3, 64, 128, 128)]
# batched vs the fp64 reference (the authoritative correctness gate): fp32 GEMM-stand-in floor.
# batched vs serial is a secondary sanity bound at the fp16 floor — the scalar serial K#1
# (run_k1_incB_host) pre-casts its GEMM operands to fp16, while the fp32 stand-in here stays
# fp32, so the two legitimately differ by ~fp16 rounding (not a transcription error).
TOL_VS_REF = 2e-3
TOL_VS_SERIAL = 2e-3


def _l2(x: torch.Tensor) -> torch.Tensor:
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def _fake_bmm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Stand-in for the batched (128,64,128) GEMM: clean fp32 ``x[Z,M,K] @ y[Z,K,N]``."""
    return x.to(torch.float32) @ y.to(torch.float32)


def _fake_mm_tc(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Stand-in for the per-chunk (128,64,128) GEMM: clean fp32 ``x[M,K] @ y[K,N]``."""
    return x.to(torch.float32) @ y.to(torch.float32)


def _fake_gemm_aa(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    """Stand-in for ``_gemm_aa``: ``out[128,64] = a[128,128] @ b[64,128]^T`` in place."""
    out.copy_((a.to(torch.float32) @ b.to(torch.float32).transpose(-1, -2)).to(out.dtype))


def _rel(got: torch.Tensor, ref: torch.Tensor) -> float:
    return ((got.float() - ref.float()).abs().max() / ref.float().abs().max().clamp_min(1e-12)).item()


def _scalar_inputs(shape: tuple[int, ...], seed: int):
    b, h, nt, c, d_k, d_v = shape
    t = nt * c
    gen = torch.Generator().manual_seed(seed)
    dt = torch.float64
    q = _l2(torch.randn(b, t, h, d_k, generator=gen, dtype=dt))
    k = _l2(torch.randn(b, t, h, d_k, generator=gen, dtype=dt))
    v = torch.randn(b, t, h, d_v, generator=gen, dtype=dt)
    g = -(torch.rand(b, t, h, generator=gen, dtype=dt) * 0.1 + 0.01)
    beta = torch.rand(b, t, h, generator=gen, dtype=dt) * 0.8 + 0.1
    do = torch.randn(b, t, h, d_v, generator=gen, dtype=dt)
    return build_microgate_bundles(q, k, v, g, beta, do, chunk_len=c, scale=d_k**-0.5)


def _cw_inputs(shape: tuple[int, ...], seed: int):
    b, h, nt, c, d_k, d_v = shape
    t = nt * c
    gen = torch.Generator().manual_seed(seed)
    dt = torch.float64
    q = _l2(torch.randn(b, t, h, d_k, generator=gen, dtype=dt))
    k = _l2(torch.randn(b, t, h, d_k, generator=gen, dtype=dt))
    v = torch.randn(b, t, h, d_v, generator=gen, dtype=dt)
    g = -(torch.rand(b, t, h, d_k, generator=gen, dtype=dt) * 0.1 + 0.01)
    bg = torch.rand(b, t, h, d_k, generator=gen, dtype=dt) * 0.8 + 0.1
    wg = torch.rand(b, t, h, d_v, generator=gen, dtype=dt) * 0.8 + 0.1
    do = torch.randn(b, t, h, d_v, generator=gen, dtype=dt)
    return build_microgate_bundles_cw(q, k, v, g, bg, wg, do, chunk_len=c, scale=d_k**-0.5)


def main() -> None:
    # Every is_available() chains back to gdn2_bwd_dhu._HAVE; the proven serial guards read
    # _HAVE directly. Flip the one root so both the _HAVE and is_available() guards pass on CPU.
    saved = {
        "have": k1mod._HAVE,
        "bmm": k1mod._bmm_tc, "gemm_aa": getattr(k1mod, "_gemm_aa", None),
        "mm": k2mod._mm_tc, "sync": torch.cuda.synchronize,
    }
    k1mod._HAVE = True
    k1mod._bmm_tc = _fake_bmm
    k1mod._gemm_aa = _fake_gemm_aa
    k2mod._mm_tc = _fake_mm_tc
    torch.cuda.synchronize = lambda *a, **kw: None
    worst_ref = 0.0
    worst_serial = 0.0
    try:
        for shape in SCALAR_SHAPES:
            bun = _scalar_inputs(shape, seed=shape[2] * 13 + 1)
            i1, e1 = bun["k1"].inputs, bun["k1"].expected
            dh, dv2, dh0 = k1mod.run_k1_incB_batched(
                i1["q"], i1["k"], i1["w"], i1["g2"], i1["g_last"], i1["do"], i1["dv_local"], i1["dht"]
            )
            s_dh, s_dv2, s_dh0 = k1mod.run_k1_incB_host(
                i1["q"], i1["k"], i1["w"], i1["g2"], i1["g_last"], i1["do"], i1["dv_local"], i1["dht"]
            )
            for got, ref in ((dh, e1["dh"]), (dv2, e1["dv2"]), (dh0, e1["dh0"])):
                worst_ref = max(worst_ref, _rel(got, ref))
            for got, ser in ((dh, s_dh), (dv2, s_dv2), (dh0, s_dh0)):
                worst_serial = max(worst_serial, _rel(got, ser))

            i2, e2 = bun["k2"].inputs, bun["k2"].expected
            kb = k2mod.run_k2_batched(
                i2["k"], i2["v"], i2["beta"], i2["g2"], i2["T"], i2["dw"], i2["du"]
            )
            ks = k2mod.run_k2_serial(
                i2["k"], i2["v"], i2["beta"], i2["g2"], i2["T"], i2["dw"], i2["du"]
            )
            for got, ref in zip(kb, (e2["dk2"], e2["dv"], e2["db"], e2["dg2"]), strict=True):
                worst_ref = max(worst_ref, _rel(got, ref))
            for got, ser in zip(kb, ks, strict=True):
                worst_serial = max(worst_serial, _rel(got, ser))
            print(f"  scalar shape={shape}: worst_ref={worst_ref:.2e} worst_serial={worst_serial:.2e}")

        for shape in CW_SHAPES:
            bun = _cw_inputs(shape, seed=shape[2] * 17 + 5)
            i1, e1 = bun["k1"].inputs, bun["k1"].expected
            dh, dv2, dh0 = k1cw.run_k1_incB_batched(
                i1["q"], i1["k"], i1["wy"], i1["g2"], i1["g_last"], i1["do"], i1["dv_local"], i1["dht"]
            )
            s_dh, s_dv2, s_dh0 = k1cw.run_k1_incB_serial(
                i1["q"], i1["k"], i1["wy"], i1["g2"], i1["g_last"], i1["do"], i1["dv_local"], i1["dht"]
            )
            for got, ref in ((dh, e1["dh"]), (dv2, e1["dv2"]), (dh0, e1["dh0"])):
                worst_ref = max(worst_ref, _rel(got, ref))
            for got, ser in ((dh, s_dh), (dv2, s_dv2), (dh0, s_dh0)):
                worst_serial = max(worst_serial, _rel(got, ser))

            i2, e2 = bun["k2"].inputs, bun["k2"].expected
            kb = k2cw.run_k2_batched(
                i2["k"], i2["v"], i2["b"], i2["w"], i2["g2"], i2["T"], i2["dwy"], i2["du"]
            )
            ks = k2cw.run_k2_serial(
                i2["k"], i2["v"], i2["b"], i2["w"], i2["g2"], i2["T"], i2["dwy"], i2["du"]
            )
            for got, ref in zip(kb, (e2["dk2"], e2["dv"], e2["db"], e2["dw"], e2["dg2"]), strict=True):
                worst_ref = max(worst_ref, _rel(got, ref))
            for got, ser in zip(kb, ks, strict=True):
                worst_serial = max(worst_serial, _rel(got, ser))
            print(f"  cw     shape={shape}: worst_ref={worst_ref:.2e} worst_serial={worst_serial:.2e}")
    finally:
        k1mod._HAVE = saved["have"]
        k1mod._bmm_tc = saved["bmm"]
        if saved["gemm_aa"] is not None:
            k1mod._gemm_aa = saved["gemm_aa"]
        k2mod._mm_tc = saved["mm"]
        torch.cuda.synchronize = saved["sync"]

    ok = worst_ref < TOL_VS_REF and worst_serial < TOL_VS_SERIAL
    print(f"\nbatched vs fp64 ref:    worst_scale_rel={worst_ref:.2e}  (tol {TOL_VS_REF:.0e})")
    print(f"batched vs serial path: worst_scale_rel={worst_serial:.2e}  (tol {TOL_VS_SERIAL:.0e})")
    print(f"GO={ok}")
    assert ok, "lever-B batching diverges from the reference or the proven serial path"


if __name__ == "__main__":
    main()
