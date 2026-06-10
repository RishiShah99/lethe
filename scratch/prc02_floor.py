"""Measure PRC-02 error floors for the scan op, to pin the gate tolerance.

Compares, at the PRC-02 shape (2, 1024, 32) with Mamba-realistic delta
(delta_bar log-uniform in [1e-3, 1e-1]):

- honest floor: fp32-accumulating scan fed fp16-rounded inputs vs the fp32
  reference fed unrounded inputs (irreducible input-rounding error)
- cheat error: fp16-accumulating scan (h and the N-dot kept in fp16)

The PRC-02 atol must sit between the two with margin.
"""

import torch
import torch.nn.functional as F


def make_inputs(b: int, length: int, d: int, n: int, seed: int = 23117):
    gen = torch.Generator().manual_seed(seed)
    u = torch.randn(b, length, d, generator=gen)
    log_lo, log_hi = torch.log(torch.tensor(1e-3)), torch.log(torch.tensor(1e-1))
    dt = torch.exp(torch.rand(b, length, d, generator=gen) * (log_hi - log_lo) + log_lo)
    delta = dt + torch.log(-torch.expm1(-dt))  # inverse softplus
    a = -torch.rand(d, n, generator=gen)
    bp = torch.randn(b, length, n, generator=gen)
    cp = torch.randn(b, length, n, generator=gen)
    ds = torch.randn(d, generator=gen)
    return u, delta, a, bp, cp, ds


def scan(u, delta, a, bp, cp, ds, acc_dtype):
    compute = torch.float32
    delta_bar = F.softplus(delta.to(compute))
    a_bar = torch.exp(delta_bar.unsqueeze(-1) * a.to(compute).unsqueeze(0).unsqueeze(0))
    b_bar = delta_bar.unsqueeze(-1) * bp.to(compute).unsqueeze(2)
    bsz, length, d = u.shape
    n = a.shape[1]
    h = torch.zeros(bsz, d, n, dtype=acc_dtype)
    ys = []
    for t in range(length):
        h = (
            a_bar[:, t].to(acc_dtype) * h
            + (b_bar[:, t] * u[:, t].to(compute).unsqueeze(-1)).to(acc_dtype)
        )
        y_t = (h * cp[:, t].to(acc_dtype).unsqueeze(1)).sum(-1) + (
            ds.to(acc_dtype) * u[:, t].to(acc_dtype)
        )
        ys.append(y_t.to(compute))
    return torch.stack(ys, dim=1)


def main() -> None:
    shape = (2, 1024, 32, 16)
    args32 = make_inputs(*shape)
    args16 = tuple(t.to(torch.float16) for t in args32)

    y_ref = scan(*args32, acc_dtype=torch.float32)
    y_honest = scan(*(t.float() for t in args16), acc_dtype=torch.float32)
    y_cheat = scan(*(t.float() for t in args16), acc_dtype=torch.float16)

    scale = y_ref.abs().max().item()
    honest = (y_honest - y_ref).abs().max().item()
    cheat = (y_cheat - y_ref).abs().max().item()
    print(f"output scale (max|y|):    {scale:.4f}")
    print(f"honest floor (fp32 acc):  {honest:.5f}")
    print(f"cheat error (fp16 acc):   {cheat:.5f}")
    print(f"separation ratio:         {cheat / honest:.1f}x")


if __name__ == "__main__":
    main()
