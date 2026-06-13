"""PRC-02 floor probe for the backward scan's grad_D view.

grad_D = sum over batch*L of dy*u — a flat ~2048-term sum whose output
scale is ~sqrt(B*L) ~ 45 at the gate shape, so both the fp16 input-
rounding floor AND one fp16 output ULP exceed the gate's flat 2e-2 atol:
the view needs the grad_A scale-aware treatment. Measures, over several
draws (the gate draws unseeded), as fractions of |ref|_inf scale:

  honest-triton : the hand-written kernel (fp32 accumulators, rounds once)
  honest-eager  : autograd in fp32 from the same fp16-rounded bits
  cheat         : grad_D re-accumulated sequentially in fp16

Usage: uv run python scratch/c2_gradd_floor.py [--device cuda] [--draws 8]
"""

from __future__ import annotations

import argparse

import torch

from flash_mamba_rl.verifier import op_harness


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--shape", default="2,1024,32")
    args = ap.parse_args()
    device = args.device
    shape = tuple(int(s) for s in args.shape.split(","))

    from flash_mamba_rl.kernels.ops.backward_selective_scan import backward_selective_scan
    from flash_mamba_rl.kernels.references.backward_selective_scan import (
        reference_backward_selective_scan,
    )

    cand = op_harness.bwd_scan_candidate_adapter(backward_selective_scan, "grad_D", saturate=False)
    ref = op_harness.bwd_scan_reference_adapter("grad_D", saturate=False)

    rows = []
    for draw in range(args.draws):
        t_fp32 = torch.randn(shape, dtype=torch.float32, device=device)
        t_fp16 = t_fp32.to(torch.float16)

        out_ref = ref(t_fp32).float()
        scale = max(1.0, out_ref.abs().max().item())

        honest_triton = (cand(t_fp16).float() - out_ref).abs().max().item() / scale

        # Eager honest: fp32 autograd from the same rounded bits, round once.
        batch, seq_len, d_model = shape
        aux16 = op_harness._bwd_scan_aux(
            batch, seq_len, d_model, op_harness.SCAN_N_STATE, t_fp16.device, torch.float16
        )
        aux32 = tuple(t.to(torch.float32) for t in aux16)
        grads32 = reference_backward_selective_scan(
            *(t.cpu() for t in aux32), t_fp16.float().cpu(), chunk_size=64
        )
        honest_eager = (
            grads32.grad_D.to(torch.float16).float().to(device) - out_ref
        ).abs().max().item() / scale

        # Cheat: the same per-element terms, accumulated sequentially in fp16.
        u16 = aux16[0]
        terms = (t_fp16 * u16).reshape(-1, d_model)
        acc = torch.zeros(d_model, dtype=torch.float16, device=t_fp16.device)
        for i in range(terms.shape[0]):
            acc = (acc + terms[i]).to(torch.float16)
        cheat = (acc.float().to(device) - out_ref).abs().max().item() / scale

        rows.append((honest_triton, honest_eager, cheat))
        print(
            f"draw {draw}: scale={scale:.1f} honest_triton={honest_triton:.3e} "
            f"honest_eager={honest_eager:.3e} cheat={cheat:.3e}",
            flush=True,
        )

    ht = max(r[0] for r in rows)
    he = max(r[1] for r in rows)
    ch = min(r[2] for r in rows)
    print(f"WORST honest_triton={ht:.3e} honest_eager={he:.3e} | BEST cheat={ch:.3e}")
    print(f"corridor: {ch / max(ht, he):.1f}x")


if __name__ == "__main__":
    main()
