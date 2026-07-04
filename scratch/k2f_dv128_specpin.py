"""fp64 ground-truth pin for the K2f dv=128 N-tiling build (desk, CPU).

The fused kernel spec ``_run_k2_fused_modelled`` is already d_v-agnostic; this confirms
it reproduces ``k2_wy_vjp_cw_ref`` to roundoff at the exact crown tile (C=64, d_k=128,
d_v=128) BEFORE the kernel is N-tiled — the mandated fp64 spec pin. No GPU needed.
"""

import sys

import torch

sys.path.insert(0, "tests")
from test_gdn2_k2_fused import _k2_bundle_inputs  # noqa: E402

from flash_mamba_rl.kernels.cute.gdn2_assemble import k2_wy_vjp_cw_ref  # noqa: E402
from flash_mamba_rl.kernels.cute.gdn2_bwd_wy_cw import _run_k2_fused_modelled  # noqa: E402

CROWN = (2, 128, 2, 128, 128, 64)  # B,T,H,d_k,d_v=128,chunk_len=64  (NT=2, crown tile)


def main() -> None:
    torch.set_default_dtype(torch.float64)
    args = _k2_bundle_inputs(CROWN, seed=11)
    got = _run_k2_fused_modelled(*args)
    exp = k2_wy_vjp_cw_ref(*args)
    names = ["dk", "dv", "db", "dw", "dg"]
    worst = 0.0
    for n, g, e in zip(names, got, exp, strict=True):
        err = (g - e).abs().max().item() / e.abs().max().clamp_min(1e-30).item()
        worst = max(worst, err)
        print(f"  {n:3s}  shape={tuple(g.shape)}  rel={err:.2e}")
    ok = worst < 1e-12
    print(f"\ncrown tile d_v=128  worst_rel={worst:.2e}  {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
