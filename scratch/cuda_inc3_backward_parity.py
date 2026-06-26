"""Inc 3 parity — CUDA selective-scan backward vs reference_backward_selective_scan.

Per-gradient scale-relative error across shapes incl. the Mamba-3 N=128 regime,
multi-chunk L, and L not a multiple of the chunk. Plus an EXC-01 check: an Inf
planted in dy must produce the SAME NaN/Inf mask as autograd (the grouping that
makes Inf*0 mint NaN where autograd does). Correctness is ground truth — speed
comes after this is green and the verifier gates pass.

    CUDA_HOME=/usr/local/cuda-13.0 PATH=$CUDA_HOME/bin:$PATH \
        CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scratch/cuda_inc3_backward_parity.py
"""

from __future__ import annotations

import torch

from flash_mamba_rl.kernels.cuda.backward import cuda_backward_scan
from flash_mamba_rl.kernels.references.backward_selective_scan import (
    reference_backward_selective_scan,
)

# (B, L, D, N) — D divisible by block_d(=8); L divisible by 64 for the reference.
SHAPES = [
    (2, 512, 256, 16),
    (2, 512, 1024, 128),  # Mamba-3 N=128
    (1, 1024, 512, 128),  # multi-chunk at items=4 (kChunk=512)
    (4, 1024, 2048, 64),
    (2, 128, 64, 128),  # short L
    (3, 320, 384, 96),  # non-pow2, L not a multiple of 512
]
FIELDS = ["grad_u", "grad_delta", "grad_A", "grad_B", "grad_C", "grad_D"]
BOUND = 3e-4


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")
    dev = torch.device("cuda")
    torch.manual_seed(0)

    all_ok = True
    worst = 0.0
    for b, length, d, n in SHAPES:
        u = torch.randn(b, length, d, device=dev)
        delta = torch.randn(b, length, d, device=dev)
        a = -torch.rand(d, n, device=dev)
        bmat = torch.randn(b, length, n, device=dev)
        cmat = torch.randn(b, length, n, device=dev)
        dskip = torch.randn(d, device=dev)
        dy = torch.randn(b, length, d, device=dev)

        ref = reference_backward_selective_scan(u, delta, a, bmat, cmat, dskip, dy, chunk_size=64)
        got = cuda_backward_scan(u, delta, a, bmat, cmat, dskip, dy)

        row = []
        for name, rg, gg in zip(FIELDS, ref, got, strict=True):
            scale = rg.abs().max().item()
            rel = (
                (gg - rg).abs().max().item() / scale if scale > 0 else (gg - rg).abs().max().item()
            )
            worst = max(worst, rel)
            ok = rel < BOUND
            all_ok &= ok
            row.append(f"{name}={rel:.2e}{'' if ok else '!!'}")
        print(f"B{b} L{length} D{d} N{n}: " + " ".join(row))

    print(f"\nworst scale-rel error {worst:.3e} (bound {BOUND:.1e})")

    # EXC-01: plant an Inf in dy, compare NaN/Inf masks against autograd.
    print("\n=== EXC-01 mask parity (Inf in dy) ===")
    torch.manual_seed(1)
    b, length, d, n = 2, 256, 256, 128
    u = torch.randn(b, length, d, device=dev)
    delta = torch.randn(b, length, d, device=dev)
    a = -torch.rand(d, n, device=dev)
    bmat = torch.randn(b, length, n, device=dev)
    cmat = torch.randn(b, length, n, device=dev)
    dskip = torch.randn(d, device=dev)
    dy = torch.randn(b, length, d, device=dev)
    dy[0, 0, 0] = float("inf")
    ref = reference_backward_selective_scan(u, delta, a, bmat, cmat, dskip, dy, chunk_size=64)
    got = cuda_backward_scan(u, delta, a, bmat, cmat, dskip, dy)
    masks_ok = True
    for name, rg, gg in zip(FIELDS, ref, got, strict=True):
        rnan, gnan = torch.isnan(rg), torch.isnan(gg)
        rinf, ginf = torch.isinf(rg), torch.isinf(gg)
        nan_match = torch.equal(rnan, gnan)
        inf_match = torch.equal(rinf, ginf)
        masks_ok &= nan_match and inf_match
        print(
            f"{name}: nan_match={nan_match} inf_match={inf_match} "
            f"(ref nan/inf={rnan.sum().item()}/{rinf.sum().item()})"
        )

    print("\nINC3-BACKWARD-OK" if (all_ok and masks_ok) else "\nINC3-BACKWARD-FAIL")


if __name__ == "__main__":
    main()
