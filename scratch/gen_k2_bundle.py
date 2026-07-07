"""Generate a K#2 (B6 WY / triangular-inverse VJP) micro-gate bundle to a .pt.

Desk side of the K#2 micro-gate: builds the bundle from the verified chunkwise
reference (``build_microgate_bundles`` → ``"k2"``) at the Phase-2 target tile and
serialises inputs + expected outputs for the on-box runner. The reference is CPU
fp64, so this is fully verifiable off-box.

K#2 inputs : k, v, beta, g2, T(=A), dw(B5), du(=dv2 from B4)
K#2 expected: dk2, dv(final), db, dg2

Usage:
    PYTHONPATH=src python scratch/gen_k2_bundle.py --nt 4 --out k2_bundle_nt4.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lethe.kernels.references.gdn2_chunkwise import build_microgate_bundles


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)


def build(b: int, h: int, nt: int, c: int, d_k: int, d_v: int, seed: int) -> dict[str, object]:
    t = nt * c
    gen = torch.Generator().manual_seed(seed)
    dt = torch.float64
    q = torch.randn(b, t, h, d_k, generator=gen, dtype=dt)
    k = torch.randn(b, t, h, d_k, generator=gen, dtype=dt)
    v = torch.randn(b, t, h, d_v, generator=gen, dtype=dt)
    g = -(torch.rand(b, t, h, generator=gen, dtype=dt) * 0.3 + 0.02)
    beta = torch.rand(b, t, h, generator=gen, dtype=dt) * 0.8 + 0.1
    do = torch.randn(b, t, h, d_v, generator=gen, dtype=dt)

    qn, kn = _l2norm(q), _l2norm(k)
    scale = d_k**-0.5
    k2 = build_microgate_bundles(
        qn, kn, v, g, beta, do, chunk_len=c, scale=scale, use_qk_l2norm=False
    )["k2"]

    payload: dict[str, object] = {
        "inputs": {name: t_.to(torch.float32) for name, t_ in k2.inputs.items()},
        "expected": {name: t_.to(torch.float32) for name, t_ in k2.expected.items()},
        "meta": {**k2.meta, "B": b, "H": h, "NT": nt, "C": c, "d_k": d_k, "d_v": d_v},
    }
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=1)
    ap.add_argument("--h", type=int, default=1)
    ap.add_argument("--nt", type=int, default=4)
    ap.add_argument("--c", type=int, default=64)
    ap.add_argument("--dk", type=int, default=128)
    ap.add_argument("--dv", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="k2_bundle.pt")
    args = ap.parse_args()

    payload = build(args.b, args.h, args.nt, args.c, args.dk, args.dv, args.seed)
    dest = Path(args.out)
    torch.save(payload, dest)

    inp = payload["inputs"]
    exp = payload["expected"]
    assert isinstance(inp, dict) and isinstance(exp, dict)
    finite = all(torch.isfinite(t_).all().item() for t_ in (*inp.values(), *exp.values()))
    print(f"wrote {dest}  finite={finite}")
    print(f"  meta={payload['meta']}")
    for name, t_ in {**inp, **exp}.items():
        assert isinstance(t_, torch.Tensor)
        print(f"  {name:6s} {tuple(t_.shape)} |{t_.abs().max().item():.4e}|max")


if __name__ == "__main__":
    main()
