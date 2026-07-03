"""Fused K#2 box GATE + RUNG-BENCH harness — kernel lives in src (gdn2_bwd_wy_f).

  --desk        CPU fp64 model gate (_run_k2_fused_modelled vs k2 bundle refs)
  (default)     box gate: run_k2_fused vs the fp64 k2 bundle at --nt/--bh/--gscale
                (GO = all 5 grads scale_rel < 5e-3 + 2-run bit-determinism)
  --drift       box gate in the training-drifted regime (--gscale 40: within-chunk
                log2 span > 128 — the c707201 NaN class; extends the drifted-regime
                regression to the fused path's fp16 mid-chain landings of dT/X)
  --bench       event-path K#2-only rung timing: fused (1 launch) vs lever-B batched
                (~10 ops incl. the 1.07 GB decay_rel build) vs per-chunk serial, with
                correctness cross-checked per candidate (a fast-but-wrong rung cannot
                pass silently).

The launcher rides torch's current stream and ends in maybe_sync (the L2/L3 capture
discipline), so gating through this harness is also the src-path gate.

    PYTHONPATH=src ~/cuteenv/bin/python scratch/k2f_microgate.py --nt 4 --out results/k2f_microgate_nt4.json
"""

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

# ─── desk gate (CPU, fp64) ────────────────────────────────────────────────────


def _l2(x: Tensor) -> Tensor:
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def _build_bundle(nt: int, b: int, h: int, gscale: float, seed: int = 42) -> dict[str, Any]:
    """In-process cw K#2 bundle at (b,h,nt,c=64,d_k=128,d_v=64); gscale drifts the gates."""
    from flash_mamba_rl.kernels.references.gdn2_chunkwise_cw import build_microgate_bundles_cw

    c, d_k, d_v = 64, 128, 64
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
    bun = build_microgate_bundles_cw(q, k, v, g, bg, wg, do, chunk_len=c, scale=d_k**-0.5)
    k2 = bun["k2"]
    return {
        "inputs": {n: tv.float() for n, tv in k2.inputs.items()},
        "expected": {n: tv.float() for n, tv in k2.expected.items()},
        "meta": {
            **k2.meta,
            "B": b,
            "H": h,
            "NT": nt,
            "C": c,
            "d_k": d_k,
            "d_v": d_v,
            "gscale": gscale,
        },
    }


def _desk_gate() -> bool:
    """CPU fp64 model check: _run_k2_fused_modelled vs k2 bundle references, tol 1e-12.

    The kernel transcribes the modelled dataflow; the fp16 landing points (dT/X round
    trips, fp16 GEMM operands) and the in-kernel fastmath exp2 are only covered by the
    silicon gate.
    """
    from flash_mamba_rl.kernels.cute.gdn2_bwd_wy_cw import _run_k2_fused_modelled
    from flash_mamba_rl.kernels.references.gdn2_chunkwise_cw import build_microgate_bundles_cw

    TOL = 1e-12
    ok = True
    for shape, gscale in [((1, 1, 1), 1.0), ((2, 2, 2), 1.0), ((1, 2, 3), 1.0), ((1, 2, 2), 40.0)]:
        b, h, nt = shape
        c, d_k, d_v = 64, 128, 64
        t = nt * c
        gen = torch.Generator().manual_seed(nt * 19 + 3)
        dt = torch.float64
        q = _l2(torch.randn(b, t, h, d_k, generator=gen, dtype=dt))
        k = _l2(torch.randn(b, t, h, d_k, generator=gen, dtype=dt))
        v = torch.randn(b, t, h, d_v, generator=gen, dtype=dt)
        g = -(torch.rand(b, t, h, d_k, generator=gen, dtype=dt) * 0.1 + 0.01) * gscale
        bg = torch.rand(b, t, h, d_k, generator=gen, dtype=dt) * 0.8 + 0.1
        wg = torch.rand(b, t, h, d_v, generator=gen, dtype=dt) * 0.8 + 0.1
        do = torch.randn(b, t, h, d_v, generator=gen, dtype=dt)
        bun = build_microgate_bundles_cw(q, k, v, g, bg, wg, do, chunk_len=c, scale=d_k**-0.5)
        inp, exp = bun["k2"].inputs, bun["k2"].expected
        mdl = _run_k2_fused_modelled(
            inp["k"], inp["v"], inp["b"], inp["w"], inp["g2"], inp["T"], inp["dwy"], inp["du"]
        )
        worst = 0.0
        for got, name in zip(mdl, ("dk2", "dv", "db", "dw", "dg2"), strict=True):
            ref = exp[name]
            r = (
                (got.double() - ref.double()).abs().max()
                / ref.double().abs().max().clamp_min(1e-12)
            ).item()
            worst = max(worst, r)
            if r > TOL:
                ok = False
        print(f"  shape={shape} gscale={gscale}: worst_rel={worst:.2e}  (tol {TOL:.0e})")
    print(f"\nDesk gate: GO={ok}")
    return ok


# ─── box gate ─────────────────────────────────────────────────────────────────


def _scale_rel(got: Tensor, exp: Tensor) -> float:
    diff = (got.float().cpu() - exp.float().cpu()).abs()
    denom = exp.float().cpu().abs().max().clamp(min=1e-12)
    return (diff.max() / denom).item()


def _compare(name: str, got: Tensor, exp: Tensor) -> dict[str, Any]:
    got_f, exp_f = got.float().cpu(), exp.float().cpu()
    diff = (got_f - exp_f).abs()
    sr = _scale_rel(got, exp)
    finite = bool(torch.isfinite(got_f).all().item())
    return {
        "name": name,
        "shape": list(got_f.shape),
        "scale_rel": sr,
        "max_abs": diff.max().item(),
        "finite": finite,
        "passed": finite and sr < 5e-3,
    }


_K2_OUT = ("dk2", "dv", "db", "dw", "dg2")


def _run_box(nt: int, b: int, h: int, gscale: float) -> dict[str, Any]:
    from flash_mamba_rl.kernels.cute.gdn2_bwd_wy_f import run_k2_fused

    payload = _build_bundle(nt, b, h, gscale)
    inp = {k_: v_.cuda() for k_, v_ in payload["inputs"].items()}
    exp = payload["expected"]
    k2_args = (
        inp["k"],
        inp["v"],
        inp["b"],
        inp["w"],
        inp["g2"],
        inp["T"],
        inp["dwy"],
        inp["du"],
    )

    got = run_k2_fused(*k2_args)
    checks = [_compare(n, g_, exp[n]) for g_, n in zip(got, _K2_OUT, strict=True)]

    got2 = run_k2_fused(*k2_args)
    det = all(torch.equal(a.cpu(), b_.cpu()) for a, b_ in zip(got, got2, strict=True))

    return {
        "device": torch.cuda.get_device_name(0),
        "meta": payload["meta"],
        "checks": checks,
        "deterministic": det,
        "GO": all(c_["passed"] for c_ in checks) and det,
    }


# ─── rung bench: fused vs lever-B batched vs per-chunk serial ─────────────────


def _run_bench(nt: int, b: int, h: int, trials: int) -> dict[str, Any]:
    from flash_mamba_rl.kernels.cute.gdn2_bwd_wy_cw import run_k2_batched, run_k2_serial
    from flash_mamba_rl.kernels.cute.gdn2_bwd_wy_f import run_k2_fused
    from flash_mamba_rl.verifier.timing import benchmark

    payload = _build_bundle(nt, b, h, 1.0)
    inp = {k_: v_.cuda() for k_, v_ in payload["inputs"].items()}
    exp = payload["expected"]
    k2_args = (
        inp["k"],
        inp["v"],
        inp["b"],
        inp["w"],
        inp["g2"],
        inp["T"],
        inp["dwy"],
        inp["du"],
    )

    # benchmark() picks CUDA-event timing from the CALL inputs; arg-less callables fall
    # to the unsynced wall path and read ~0 for async work (the burst-2 v2 artifact).
    sync_probe = torch.zeros(1, device="cuda")

    def bench_cuda(fn: Any) -> float:
        return benchmark(lambda _p: fn(), (sync_probe,), warmup=5, trials=trials).median_ms

    candidates: dict[str, Any] = {
        "k2_fused": lambda: run_k2_fused(*k2_args),
        "k2_batched": lambda: run_k2_batched(*k2_args),
    }
    if nt * b * h <= 64:  # the per-chunk loop is O(Z) launches — cap the slow rung
        candidates["k2_serial"] = lambda: run_k2_serial(*k2_args)

    row: dict[str, Any] = {
        "shape": {"B": b, "H": h, "NT": nt, "L": nt * 64, "c": 64, "d_k": 128, "d_v": 64},
        "trials": trials,
    }
    for name, fn in candidates.items():
        try:
            got = fn()  # warm/JIT + correctness cross-check
            worst = max(_scale_rel(g_, exp[n]) for g_, n in zip(got, _K2_OUT, strict=True))
            row[f"{name}_scale_rel"] = worst
            row[f"{name}_ms"] = bench_cuda(fn)
        except Exception:
            row[f"{name}_err"] = traceback.format_exc()
        print(
            f"  {name}: ms={row.get(f'{name}_ms')} scale_rel={row.get(f'{name}_scale_rel')}",
            flush=True,
        )

    if "k2_batched_ms" in row and row.get("k2_fused_ms"):
        row["batched_over_fused"] = row["k2_batched_ms"] / row["k2_fused_ms"]
    return row


# ─── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    # Parse before cutlass import-time argparse fires (tmem_offset_probe.py pattern).
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--desk", action="store_true")
    ap.add_argument("--drift", action="store_true", help="box gate at --gscale 40 (drifted regime)")
    ap.add_argument("--bench", action="store_true", help="rung bench: fused vs batched vs serial")
    ap.add_argument("--nt", type=int, default=4)
    ap.add_argument("--bh", type=str, default="2,2", help="bundle batch,heads")
    ap.add_argument("--gscale", type=float, default=1.0)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("-h", "--help", action="help")
    args = ap.parse_args()

    b, h = (int(x) for x in args.bh.split(","))
    res: dict[str, Any] = {"nt": args.nt, "B": b, "H": h}
    try:
        if args.desk:
            res["GO"] = _desk_gate()
        elif args.bench:
            if not torch.cuda.is_available():
                raise RuntimeError("--bench needs CUDA (box)")
            res.update(_run_bench(args.nt, b, h, args.trials))
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("box gate needs CUDA (sm_100 box)")
            gscale = 40.0 if args.drift else args.gscale
            res.update(_run_box(args.nt, b, h, gscale))
    except Exception as exc:
        res["error"] = f"{type(exc).__name__}: {exc}"
        res["trace"] = traceback.format_exc()
        res["GO"] = False

    if args.out:
        dest = Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str))
    print(f"\nGO={res.get('GO')}")


if __name__ == "__main__":
    main()
