"""Box micro-gate for the K2f dv=128 N-tiling (crown). Runs the fused kernel at
d_v=128 vs the fp64 k2 bundle; GO = all 5 grads scale_rel < 5e-3 + bit-determinism.

  PYTHONPATH=src ~/cuteenv/bin/python scratch/k2f_dv128_gate.py --nt 4
  PYTHONPATH=src ~/cuteenv/bin/python scratch/k2f_dv128_gate.py --nt 4 --gscale 40   # drift
"""

import argparse
import json
import traceback
from pathlib import Path

import torch
from torch import Tensor

_K2_OUT = ("dk2", "dv", "db", "dw", "dg2")


def _l2(x: Tensor) -> Tensor:
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def _scale_rel(got: Tensor, exp: Tensor) -> float:
    return (
        (got.float() - exp.float()).abs().max() / exp.float().abs().max().clamp_min(1e-12)
    ).item()


def _bundle(nt, b, h, gscale, d_v=128, seed=42):
    from lethe.kernels.references.gdn2_chunkwise_cw import build_microgate_bundles_cw

    c, d_k = 64, 128
    t = nt * c
    gen = torch.Generator().manual_seed(seed)
    dt = torch.float64
    q = _l2(torch.randn(b, t, h, d_k, generator=gen, dtype=dt))
    k = _l2(torch.randn(b, t, h, d_k, generator=gen, dtype=dt))
    v = torch.randn(b, t, h, d_v, generator=gen, dtype=dt)
    g = -(torch.rand(b, t, h, d_k, generator=gen, dtype=dt) * 0.1 + 0.01) * gscale
    bg = torch.rand(b, t, h, d_k, generator=gen, dtype=dt) * 0.8 + 0.1
    wg = torch.rand(b, t, h, d_v, generator=gen, dtype=dt) * 0.8 + 0.1
    do = torch.randn(b, t, h, d_v, generator=gen, dtype=dt)
    k2 = build_microgate_bundles_cw(q, k, v, g, bg, wg, do, chunk_len=c, scale=d_k**-0.5)["k2"]
    return {n: tv.float() for n, tv in k2.inputs.items()}, {
        n: tv.float() for n, tv in k2.expected.items()
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nt", type=int, default=4)
    ap.add_argument("--bh", type=int, nargs=2, default=[2, 8])
    ap.add_argument("--gscale", type=float, default=1.0)
    ap.add_argument("--out", type=str, default="")
    args, _ = ap.parse_known_args()

    if not torch.cuda.is_available():
        print("no CUDA (desk)")
        return
    from lethe.kernels.cute.gdn2_bwd_wy_f import run_k2_fused

    b, h = args.bh
    inp, exp = _bundle(args.nt, b, h, args.gscale)
    inp = {kk: vv.cuda() for kk, vv in inp.items()}
    exp = {kk: vv.cuda() for kk, vv in exp.items()}
    a = (inp["k"], inp["v"], inp["b"], inp["w"], inp["g2"], inp["T"], inp["dwy"], inp["du"])

    res = {"nt": args.nt, "bh": args.bh, "gscale": args.gscale, "d_v": 128}
    try:
        got = run_k2_fused(*a)
        got2 = run_k2_fused(*a)
        srs = {n: _scale_rel(g_, exp[n]) for g_, n in zip(got, _K2_OUT, strict=True)}
        det = all(torch.equal(g1, g2) for g1, g2 in zip(got, got2, strict=True))
        worst = max(srs.values())
        res["scale_rel"] = srs
        res["worst_scale_rel"] = worst
        res["deterministic"] = det
        res["shapes"] = {n: list(g_.shape) for g_, n in zip(got, _K2_OUT, strict=True)}
        res["GO"] = bool(worst < 5e-3 and det)
    except Exception:
        res["err"] = traceback.format_exc()
        res["GO"] = False

    print(json.dumps(res, indent=2, default=str))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2, default=str))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
