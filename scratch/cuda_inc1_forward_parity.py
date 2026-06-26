"""Inc 1 parity — CUDA forward selective scan vs reference_forward_chunked_scan.

Confirms the cub::BlockScan forward kernel matches the fp32 oracle across
shapes incl. the Mamba-3 N=128 regime and L not a multiple of blockDim (the
chunk-carry path). Scale-relative bound mirrors the C1 forward parity budget
(reduction-order noise ~ eps*sqrt(chain)*scale).

    CUDA_HOME=/usr/local/cuda-13.0 PATH=$CUDA_HOME/bin:$PATH \
        CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scratch/cuda_inc1_forward_parity.py
"""

from __future__ import annotations

import torch

from flash_mamba_rl.kernels.cuda.forward import cuda_forward_scan
from flash_mamba_rl.kernels.references.forward_chunked_scan import reference_forward_chunked_scan

# (B, L, D, N) — L divisible by 64 for the reference's chunking.
SHAPES = [
    (2, 512, 256, 16),
    (2, 512, 1024, 128),  # Mamba-3 N=128
    (1, 2048, 512, 128),  # long-ish L, chunk-carry exercised
    (4, 1024, 2048, 64),
    (2, 128, 64, 128),  # short L (< blockDim), N > L
    (3, 320, 384, 96),  # non-pow2 everything, L not a multiple of 256
]

BOUND = 2e-4  # scale-relative


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")
    dev = torch.device("cuda")
    torch.manual_seed(0)

    worst = 0.0
    all_ok = True
    for b, length, d, n in SHAPES:
        u = torch.randn(b, length, d, device=dev)
        delta = torch.randn(b, length, d, device=dev)
        a = -torch.rand(d, n, device=dev)
        bmat = torch.randn(b, length, n, device=dev)
        cmat = torch.randn(b, length, n, device=dev)
        dskip = torch.randn(d, device=dev)

        y_ref = reference_forward_chunked_scan(u, delta, a, bmat, cmat, dskip, chunk_size=64)
        y_cuda = cuda_forward_scan(u, delta, a, bmat, cmat, dskip)

        scale = y_ref.abs().max().item()
        max_abs = (y_cuda - y_ref).abs().max().item()
        rel = max_abs / scale if scale > 0 else max_abs
        worst = max(worst, rel)
        ok = rel < BOUND
        all_ok &= ok
        print(
            f"B{b} L{length} D{d} N{n}: max_abs={max_abs:.3e} scale={scale:.3e} "
            f"rel={rel:.3e} {'OK' if ok else 'FAIL'}"
        )

    print(f"\nworst scale-rel error {worst:.3e} (bound {BOUND:.1e})")
    print("INC1-FORWARD-OK" if all_ok else "INC1-FORWARD-FAIL")


if __name__ == "__main__":
    main()
