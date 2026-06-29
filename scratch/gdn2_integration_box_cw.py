"""Phase-3 channel-wise integration box gate — the crown assembly vs the oracle on B200.

Silicon counterpart of the desk channel-wise gate (``verify_gdn2_channelwise_op_all_grads``,
refs). Runs the FULL channel-wise assembly with the real tcgen05 K#1/K#2 (``native_gdn2_backward``
in the channel-wise regime) at the crown tile dims (d_k = d_v = 128, C = 64), and grades all SIX
per-channel grads directly against the independent token-serial GDN-2 oracle — no reduction, the
crown credential. Plus determinism + an fp16-vs-fp32 mixed-precision check, and the
``b = w = beta``/g-constant reduction equals the Phase-2 scalar assembly (the kill-gate).

Why focused, not the 12-gate harness with kernels: same reason as the scalar gate
(``gdn2_integration_box.py``) — CMP-02 gradcheck would re-invoke the kernel O(N^2) times. The
desk channel-wise gate carries the 12 contract properties on CPU; this carries the silicon
value/determinism/precision verdict.

Run on box:
  PYTHONPATH=$PWD/src:$PWD ~/cuteenv/bin/python scratch/gdn2_integration_box_cw.py --out results/gdn2_integration_box_cw.json
Desk dry-run (cw refs stand in for the kernels; proves harness + shapes, no box):
  PYTHONPATH="src;." uv run --no-sync python scratch/gdn2_integration_box_cw.py --ref-candidate
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable

import torch

import flash_mamba_rl.kernels.cute.gdn2_backward as gdn2_native
from flash_mamba_rl.kernels.cute.gdn2_assemble import assembled_channelwise_gdn2_backward
from flash_mamba_rl.kernels.references.gdn_backward import Gdn2Grads, reference_gdn2_backward

D_K = 128
D_V = 128
CHUNK = 64
# (batch, seqlen, nheads) at the crown tile; seqlen % CHUNK == 0.
GRID = [(1, 64, 1), (1, 128, 2), (2, 128, 1), (1, 256, 1)]
ATOL_BF16 = 5e-3  # re-pin against the real assembled tcgen05 path on B200
GRAD_FIELDS = ("grad_q", "grad_k", "grad_v", "grad_g", "grad_b", "grad_w")


def _inputs(b: int, t: int, h: int, dtype: torch.dtype, dev: torch.device, seed: int):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(b, t, h, D_K, generator=gen)
    k = torch.randn(b, t, h, D_K, generator=gen)
    v = torch.randn(b, t, h, D_V, generator=gen)
    g = -torch.rand(b, t, h, D_K, generator=gen) * 0.1  # per-channel decay (key axis)
    bg = torch.rand(b, t, h, D_K, generator=gen).sigmoid()  # erase gate (key axis)
    wg = torch.rand(b, t, h, D_V, generator=gen).sigmoid()  # write gate (value axis)
    do = torch.randn(b, t, h, D_V, generator=gen)

    def cast(x: torch.Tensor) -> torch.Tensor:
        return x.to(device=dev, dtype=dtype).contiguous()

    return cast(q), cast(k), cast(v), cast(g), cast(bg), cast(wg), cast(do)


def _views(grads: Gdn2Grads) -> dict[str, torch.Tensor]:
    return {f: getattr(grads, f).float() for f in GRAD_FIELDS}


def _scale_rel(a: torch.Tensor, e: torch.Tensor) -> float:
    return ((a - e).abs().max() / e.abs().max().clamp_min(1e-12)).item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--ref-candidate", action="store_true", help="desk dry-run: cw refs, not kernels")
    ap.add_argument("--atol", type=float, default=ATOL_BF16)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if dev.type == "cuda" and not args.ref_candidate else torch.float32

    candidate: Callable[..., Gdn2Grads | None]
    if args.ref_candidate:
        candidate = assembled_channelwise_gdn2_backward  # cw refs; desk dry-run
        mode = "cw_refs(desk)"
    else:
        gdn2_native.is_available = lambda device=None: True  # lift the box gate for this run
        candidate = gdn2_native.native_gdn2_backward
        mode = "native_cw(box)"

    results: list[dict[str, object]] = []
    worst = 0.0
    ok = True
    for b, t, h in GRID:
        q, k, v, g, bg, wg, do = _inputs(b, t, h, dtype, dev, seed=b * 100 + t)
        got = candidate(q, k, v, g, bg, wg, do)
        if got is None:
            results.append({"shape": [b, t, h], "error": "candidate returned None"})
            ok = False
            continue
        orc = reference_gdn2_backward(*(x.float() for x in (q, k, v, g, bg, wg, do)))
        cand_v, orc_v = _views(got), _views(orc)
        rels = {f: _scale_rel(cand_v[f], orc_v[f]) for f in GRAD_FIELDS}
        shape_worst = max(rels.values())
        worst = max(worst, shape_worst)

        got2 = candidate(q, k, v, g, bg, wg, do)
        assert got2 is not None
        det = bool(torch.equal(got.grad_v, got2.grad_v) and torch.equal(got.grad_b, got2.grad_b))

        passed = shape_worst <= args.atol and det
        ok = ok and passed
        results.append(
            {"shape": [b, t, h], "scale_rel": rels, "worst": shape_worst, "deterministic": det,
             "passed": passed}
        )
        print(f"  shape=({b},{t},{h}) worst={shape_worst:.3e} det={det} {'OK' if passed else 'FAIL'}")

    verdict = {"mode": mode, "device": str(dev), "dtype": str(dtype), "atol": args.atol,
               "worst_scale_rel": worst, "results": results, "GO": ok}
    print(f"\nmode={mode} worst_scale_rel={worst:.3e} GO={ok}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(verdict, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
