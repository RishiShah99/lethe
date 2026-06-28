"""Generate a K#1 (B4 reverse-state scan) micro-gate bundle and save it to a .pt.

Desk side of the K#1 micro-gate: builds the bundle from the verified chunkwise
reference (``references/gdn2_chunkwise.build_microgate_bundles``), at the Phase-2
target tile (C=64, d_k=128, d_v=64) by default, and serialises inputs + expected
outputs to a .pt the on-box runner (``scratch/k1_microgate.py``) loads. The reference
runs on CPU in fp64, so this is fully verifiable off-box.

Usage:
    PYTHONPATH=src python scratch/gen_k1_bundle.py --nt 1 --out k1_bundle_nt1.pt
    PYTHONPATH=src python scratch/gen_k1_bundle.py --nt 4 --out k1_bundle_nt4.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from flash_mamba_rl.kernels.references.gdn2_chunkwise import build_microgate_bundles


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)


def build(
    b: int, h: int, nt: int, c: int, d_k: int, d_v: int, seed: int
) -> dict[str, object]:
    t = nt * c
    gen = torch.Generator().manual_seed(seed)
    dt = torch.float64
    q = torch.randn(b, t, h, d_k, generator=gen, dtype=dt)
    k = torch.randn(b, t, h, d_k, generator=gen, dtype=dt)
    v = torch.randn(b, t, h, d_v, generator=gen, dtype=dt)
    g = -(torch.rand(b, t, h, generator=gen, dtype=dt) * 0.3 + 0.02)  # log-decay < 0
    beta = torch.rand(b, t, h, generator=gen, dtype=dt) * 0.8 + 0.1
    do = torch.randn(b, t, h, d_v, generator=gen, dtype=dt)

    qn, kn = _l2norm(q), _l2norm(k)
    scale = d_k**-0.5
    bundles = build_microgate_bundles(
        qn, kn, v, g, beta, do, chunk_len=c, scale=scale, use_qk_l2norm=False
    )
    k1 = bundles["k1"]

    # store fp32 (the kernel runs bf16 I/O + fp32 accum; expected stays fp32 for the
    # tolerance compare). head-major chunked shapes [B,H,NT,...].
    payload: dict[str, object] = {
        "inputs": {name: t_.to(torch.float32) for name, t_ in k1.inputs.items()},
        "expected": {name: t_.to(torch.float32) for name, t_ in k1.expected.items()},
        "meta": {**k1.meta, "B": b, "H": h, "NT": nt, "C": c, "d_k": d_k, "d_v": d_v},
    }
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=1)
    ap.add_argument("--h", type=int, default=1)
    ap.add_argument("--nt", type=int, default=1)
    ap.add_argument("--c", type=int, default=64)
    ap.add_argument("--dk", type=int, default=128)
    ap.add_argument("--dv", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="k1_bundle.pt")
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
        print(f"  {name:9s} {tuple(t_.shape)} |{t_.abs().max().item():.4e}|max")


if __name__ == "__main__":
    main()
