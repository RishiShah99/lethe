"""Desk gate for lever D — inc-B2's flat-L marshalling + b_ga round-trip (scalar + cw) on CPU.

inc-B2 fuses the reverse it-loop into one persistent kernel; the GEMMs are box-only, but the
new host marshalling (the flat-L = n_bh·NT operand stacking, the do^T prefill, and the b_dv^T
round-trip into b_ga's second half) is pure torch. ``_run_k1_incB2_modelled`` runs that exact
dataflow with the two ``@`` standing in for the kernel's tcgen05 GEMMs (out = a @ b^T). This
grades it against (a) the fp64 micro-gate references and (b) the box-proven host reverse loop
(``run_k1_incB_host`` / ``run_k1_incB_serial``). If the flat-L packing or the round-trip layout
is wrong it shows here for free; the box then only confirms the DSL transcription (in-kernel
loop, dynamic-chunk TMA, the GMEM round-trip fence) — the GEMM itself is the proven config.

    PYTHONPATH="src;." uv run --no-sync python scratch/k1_incB2_orchestration_check.py
"""

from __future__ import annotations

import torch

import lethe.kernels.cute.gdn2_bwd_dhu as k1mod
import lethe.kernels.cute.gdn2_bwd_dhu_cw as k1cw
import lethe.kernels.cute.gdn2_bwd_wy as k2mod
from lethe.kernels.references.gdn2_chunkwise import build_microgate_bundles
from lethe.kernels.references.gdn2_chunkwise_cw import build_microgate_bundles_cw

# (b, h, nt, c, d_k, d_v) — gentle decays so the fp32 GEMM stand-in stays near the fp64 ref.
SCALAR_SHAPES = [(1, 1, 1, 64, 128, 64), (2, 2, 4, 64, 128, 64), (1, 1, 8, 64, 128, 64)]
CW_SHAPES = [(1, 1, 1, 64, 128, 128), (2, 2, 2, 64, 128, 64), (1, 1, 3, 64, 128, 128)]
TOL_VS_REF = 2e-3  # modelled (fp32 GEMM stand-in) vs the fp64 micro-gate reference
TOL_VS_HOST = 2e-3  # modelled vs the box-proven host reverse loop (also fp32-swapped)


def _l2(x: torch.Tensor) -> torch.Tensor:
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def _fake_gemm_aa(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    """Stand-in for ``_gemm_aa``: ``out[128,64] = a[128,128] @ b[64,128]^T`` in place."""
    out.copy_((a.to(torch.float32) @ b.to(torch.float32).transpose(-1, -2)).to(out.dtype))


def _fake_mm_tc(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Stand-in for the per-chunk (128,64,128) GEMM: clean fp32 ``x[M,K] @ y[K,N]``."""
    return x.to(torch.float32) @ y.to(torch.float32)


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
    # Flip the one _HAVE root + swap the GEMM helpers so the proven host path runs in fp32 on CPU.
    saved = {
        "have": k1mod._HAVE,
        "gemm_aa": getattr(k1mod, "_gemm_aa", None),
        "sync": torch.cuda.synchronize,
        "mm": k2mod._mm_tc,  # cw host loop routes through _mm_tc
    }
    k1mod._HAVE = True
    k1mod._gemm_aa = _fake_gemm_aa
    k2mod._mm_tc = _fake_mm_tc
    torch.cuda.synchronize = lambda *a, **kw: None
    worst_ref = 0.0
    worst_host = 0.0
    try:
        for shape in SCALAR_SHAPES:
            bun = _scalar_inputs(shape, seed=shape[2] * 13 + 1)
            i1, e1 = bun["k1"].inputs, bun["k1"].expected
            mdl = k1mod._run_k1_incB2_modelled(
                i1["q"], i1["k"], i1["w"], i1["g2"], i1["g_last"], i1["do"], i1["dv_local"], i1["dht"]
            )
            host = k1mod.run_k1_incB_host(
                i1["q"], i1["k"], i1["w"], i1["g2"], i1["g_last"], i1["do"], i1["dv_local"], i1["dht"]
            )
            for got, ref in zip(mdl, (e1["dh"], e1["dv2"], e1["dh0"]), strict=True):
                worst_ref = max(worst_ref, _rel(got, ref))
            for got, hst in zip(mdl, host, strict=True):
                worst_host = max(worst_host, _rel(got, hst))
            print(f"  scalar shape={shape}: worst_ref={worst_ref:.2e} worst_host={worst_host:.2e}")

        for shape in CW_SHAPES:
            bun = _cw_inputs(shape, seed=shape[2] * 17 + 5)
            i1, e1 = bun["k1"].inputs, bun["k1"].expected
            mdl = k1cw._run_k1_incB2_modelled(
                i1["q"], i1["k"], i1["wy"], i1["g2"], i1["g_last"], i1["do"], i1["dv_local"], i1["dht"]
            )
            host = k1cw.run_k1_incB_serial(
                i1["q"], i1["k"], i1["wy"], i1["g2"], i1["g_last"], i1["do"], i1["dv_local"], i1["dht"]
            )
            for got, ref in zip(mdl, (e1["dh"], e1["dv2"], e1["dh0"]), strict=True):
                worst_ref = max(worst_ref, _rel(got, ref))
            for got, hst in zip(mdl, host, strict=True):
                worst_host = max(worst_host, _rel(got, hst))
            print(f"  cw     shape={shape}: worst_ref={worst_ref:.2e} worst_host={worst_host:.2e}")
    finally:
        k1mod._HAVE = saved["have"]
        if saved["gemm_aa"] is not None:
            k1mod._gemm_aa = saved["gemm_aa"]
        k2mod._mm_tc = saved["mm"]
        torch.cuda.synchronize = saved["sync"]

    ok = worst_ref < TOL_VS_REF and worst_host < TOL_VS_HOST
    print(f"\ninc-B2 modelled vs fp64 ref:  worst_scale_rel={worst_ref:.2e}  (tol {TOL_VS_REF:.0e})")
    print(f"inc-B2 modelled vs host loop: worst_scale_rel={worst_host:.2e}  (tol {TOL_VS_HOST:.0e})")
    print(f"GO={ok}")
    assert ok, "inc-B2 flat-L marshalling diverges from the reference or the proven host loop"


if __name__ == "__main__":
    main()
