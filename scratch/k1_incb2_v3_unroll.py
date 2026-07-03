"""Level-3 fused K#1 box GATE + RUNG-BENCH harness — kernel lives in src (wired at burst-4).

The unrolled fused reverse-scan kernel + launcher were moved to
``flash_mamba_rl.kernels.cute.gdn2_bwd_dhu_l3`` after the burst-3 silicon gates
(results/k1_incb2_v3_{scalar_nt1,scalar_nt4,cw_nt4,cw_nt8}.json — GO + deterministic,
worst scale_rel ~6.5e-4; design rationale in the module docstring). This file is the
reproducibility layer:

  --desk        CPU fp64 model gate (_run_k1_incB2_modelled vs bundle refs, scalar + cw)
  (default)     box gate: run_k1_incB2_v3 vs the fp64 bundle at --mode/--nt
                (GO = all scale_rel < 5e-3 + 2-run bit-determinism)
  --bench       THE RUNG BENCH (cw only): event-path K#1-only timing of the ladder —
                L3 fused (1 launch) vs L2 epilogue-fused (2/chunk) vs lever-B batched
                (~8 ops/chunk) on the same bundle inputs. Answers whether fusing the
                reverse loop closes the per-chunk launch/glue gap at the kernel rung.

Since the launcher now rides torch's current stream and ends in maybe_sync (the L2
capture discipline), gating through THIS harness is also the src-path re-gate.
"""

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from flash_mamba_rl.kernels.cute.gdn2_bwd_dhu_l3 import run_k1_incB2_v3

# ─── desk gate (CPU, fp64) ────────────────────────────────────────────────────


def _desk_gate() -> bool:
    """CPU fp64 model check: _run_k1_incB2_modelled vs bundle references (scalar + cw).

    Mirrors scratch/k1_incB2_orchestration_check.py; target ≤ 1e-12 relative error.
    The kernel transcribes the same dataflow as the modelled specs (only the lifecycle
    differs), so they remain the ground truth. cw shapes are d_v=64 only (single-N-tile
    envelope). The kernel's fp16 landing points — m_bdv (G1 TMEM→GMEM epilogue),
    dv2/b_ga[:,:,C:] (SIMT glue), m_t (GA epilogue), b_dhT (carry round-trip) — are only
    covered by the silicon gate.
    """
    import flash_mamba_rl.kernels.cute.gdn2_bwd_dhu as k1mod
    import flash_mamba_rl.kernels.cute.gdn2_bwd_dhu_cw as k1cw
    from flash_mamba_rl.kernels.references.gdn2_chunkwise import build_microgate_bundles
    from flash_mamba_rl.kernels.references.gdn2_chunkwise_cw import build_microgate_bundles_cw

    TOL = 1e-12

    def _l2(x: Tensor) -> Tensor:
        return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)

    def _rel(got: Tensor, ref: Tensor) -> float:
        return (
            (got.double() - ref.double()).abs().max()
            / ref.double().abs().max().clamp_min(1e-12)
        ).item()

    scalar_shapes = [(1, 1, 1, 64, 128, 64), (2, 2, 4, 64, 128, 64), (1, 1, 8, 64, 128, 64)]
    # d_v=64 only: single-N-tile envelope; d_v=128 belongs to the N-tiling increment.
    cw_shapes = [(1, 1, 1, 64, 128, 64), (2, 2, 2, 64, 128, 64), (1, 1, 3, 64, 128, 64)]

    ok = True

    for shape in scalar_shapes:
        b, h, nt, c, d_k, d_v = shape
        t = nt * c
        gen = torch.Generator().manual_seed(nt * 13 + 1)
        dt = torch.float64
        q = _l2(torch.randn(b, t, h, d_k, generator=gen, dtype=dt))
        k_t = _l2(torch.randn(b, t, h, d_k, generator=gen, dtype=dt))
        v = torch.randn(b, t, h, d_v, generator=gen, dtype=dt)
        g = -(torch.rand(b, t, h, generator=gen, dtype=dt) * 0.1 + 0.01)
        beta = torch.rand(b, t, h, generator=gen, dtype=dt) * 0.8 + 0.1
        do = torch.randn(b, t, h, d_v, generator=gen, dtype=dt)
        bun = build_microgate_bundles(q, k_t, v, g, beta, do, chunk_len=c, scale=d_k**-0.5)
        i1, e1 = bun["k1"].inputs, bun["k1"].expected
        mdl = k1mod._run_k1_incB2_modelled(
            i1["q"], i1["k"], i1["w"], i1["g2"], i1["g_last"],
            i1["do"], i1["dv_local"], i1["dht"],
        )
        worst = 0.0
        for got, ref in zip(mdl, (e1["dh"], e1["dv2"], e1["dh0"]), strict=True):
            r = _rel(got, ref)
            worst = max(worst, r)
            if r > TOL:
                ok = False
        print(f"  scalar shape={shape}: worst_rel={worst:.2e}  (tol {TOL:.0e})")

    for shape in cw_shapes:
        b, h, nt, c, d_k, d_v = shape
        t = nt * c
        gen = torch.Generator().manual_seed(nt * 17 + 5)
        dt = torch.float64
        q = _l2(torch.randn(b, t, h, d_k, generator=gen, dtype=dt))
        k_t = _l2(torch.randn(b, t, h, d_k, generator=gen, dtype=dt))
        v = torch.randn(b, t, h, d_v, generator=gen, dtype=dt)
        g = -(torch.rand(b, t, h, d_k, generator=gen, dtype=dt) * 0.1 + 0.01)
        bg = torch.rand(b, t, h, d_k, generator=gen, dtype=dt) * 0.8 + 0.1
        wg = torch.rand(b, t, h, d_v, generator=gen, dtype=dt) * 0.8 + 0.1
        do = torch.randn(b, t, h, d_v, generator=gen, dtype=dt)
        bun = build_microgate_bundles_cw(q, k_t, v, g, bg, wg, do, chunk_len=c, scale=d_k**-0.5)
        i1, e1 = bun["k1"].inputs, bun["k1"].expected
        mdl = k1cw._run_k1_incB2_modelled(
            i1["q"], i1["k"], i1["wy"], i1["g2"], i1["g_last"],
            i1["do"], i1["dv_local"], i1["dht"],
        )
        worst = 0.0
        for got, ref in zip(mdl, (e1["dh"], e1["dv2"], e1["dh0"]), strict=True):
            r = _rel(got, ref)
            worst = max(worst, r)
            if r > TOL:
                ok = False
        print(f"  cw     shape={shape}: worst_rel={worst:.2e}  (tol {TOL:.0e})")

    print(f"\nDesk gate: GO={ok}  (tol {TOL:.0e})")
    return ok


# ─── box harness ──────────────────────────────────────────────────────────────


def _scale_rel(got: Tensor, exp: Tensor) -> float:
    """scale_rel = diff.max() / exp.abs().max().clamp_min(1e-12) — Phase-2/3 GO ledger metric."""
    diff = (got.float().cpu() - exp.float().cpu()).abs()
    denom = exp.float().cpu().abs().max().clamp(min=1e-12)
    return (diff.max() / denom).item()


def _compare(name: str, got: Tensor, exp: Tensor) -> dict[str, Any]:
    got_f = got.float().cpu()
    exp_f = exp.float().cpu()
    diff = (got_f - exp_f).abs()
    sr = _scale_rel(got, exp)
    max_abs = diff.max().item()
    max_rel = (diff / exp_f.abs().clamp_min(1e-6)).max().item()
    finite = bool(torch.isfinite(got_f).all().item())
    # GO criterion: scale_rel < 5e-3 (Phase-2/3 ledger; 3.29e-3/3.31e-3 prior marks).
    passed = finite and sr < 5e-3
    return {
        "name": name,
        "shape": list(got_f.shape),
        "scale_rel": sr,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "finite": finite,
        "passed": passed,
    }


def _build_bundle(mode: str, nt: int = 4, b: int = 2, h: int = 2) -> dict[str, Any]:
    """Build an in-process K#1 bundle at (b,h,nt,c=64,d_k=128,d_v=64); nt is the stage knob."""
    from flash_mamba_rl.kernels.references.gdn2_chunkwise import build_microgate_bundles
    from flash_mamba_rl.kernels.references.gdn2_chunkwise_cw import build_microgate_bundles_cw

    c, d_k, d_v = 64, 128, 64
    t = nt * c
    gen = torch.Generator().manual_seed(42)
    dt = torch.float64

    def _l2(x: Tensor) -> Tensor:
        return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)

    q = _l2(torch.randn(b, t, h, d_k, generator=gen, dtype=dt))
    k_t = _l2(torch.randn(b, t, h, d_k, generator=gen, dtype=dt))
    v = torch.randn(b, t, h, d_v, generator=gen, dtype=dt)
    do = torch.randn(b, t, h, d_v, generator=gen, dtype=dt)

    if mode == "cw":
        g = -(torch.rand(b, t, h, d_k, generator=gen, dtype=dt) * 0.1 + 0.01)
        bg = torch.rand(b, t, h, d_k, generator=gen, dtype=dt) * 0.8 + 0.1
        wg = torch.rand(b, t, h, d_v, generator=gen, dtype=dt) * 0.8 + 0.1
        bun = build_microgate_bundles_cw(q, k_t, v, g, bg, wg, do, chunk_len=c, scale=d_k**-0.5)
    else:
        g = -(torch.rand(b, t, h, generator=gen, dtype=dt) * 0.1 + 0.01)
        beta = torch.rand(b, t, h, generator=gen, dtype=dt) * 0.8 + 0.1
        bun = build_microgate_bundles(q, k_t, v, g, beta, do, chunk_len=c, scale=d_k**-0.5)

    k1 = bun["k1"]
    return {
        "inputs": {n: tv.float() for n, tv in k1.inputs.items()},
        "expected": {n: tv.float() for n, tv in k1.expected.items()},
        "meta": {**k1.meta, "B": b, "H": h, "NT": nt, "C": c, "d_k": d_k, "d_v": d_v},
    }


def _run_box(bundle_path: str, mode: str, nt: int) -> dict[str, Any]:
    bp = Path(bundle_path) if bundle_path else None
    # NB: Path("") normalizes to "." (exists!) — never Path() an empty string here.
    if bp is None or not bp.exists():
        print(f"{'--bundle absent' if not bundle_path else bp} — building in-process "
              f"(b=2,h=2,nt={nt},c=64,d_k=128,d_v=64, mode={mode})")
        payload = _build_bundle(mode, nt=nt)
        if bundle_path:
            bp.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, bp)
            print(f"saved {bp}")
    else:
        payload = torch.load(bundle_path, weights_only=False)

    inp = {k: v.cuda() for k, v in payload["inputs"].items()}
    exp = payload["expected"]
    cw = mode == "cw"
    key_w = "wy" if cw else "w"

    dh, dv2, dh0 = run_k1_incB2_v3(
        inp["q"], inp["k"], inp[key_w], inp["g2"], inp["g_last"],
        inp["do"], inp["dv_local"], inp["dht"], cw=cw,
    )
    checks = [
        _compare("dh", dh, exp["dh"]),
        _compare("dv2", dv2, exp["dv2"]),
        _compare("dh0", dh0, exp["dh0"]),
    ]

    # Second run for determinism (bit-exact).
    dh2, dv2_2, dh0_2 = run_k1_incB2_v3(
        inp["q"], inp["k"], inp[key_w], inp["g2"], inp["g_last"],
        inp["do"], inp["dv_local"], inp["dht"], cw=cw,
    )
    det = (
        torch.equal(dh.cpu(), dh2.cpu())
        and torch.equal(dv2.cpu(), dv2_2.cpu())
        and torch.equal(dh0.cpu(), dh0_2.cpu())
    )

    return {
        "device": torch.cuda.get_device_name(0),
        "bundle": bundle_path,
        "mode": mode,
        "meta": payload.get("meta", {}),
        "checks": checks,
        "deterministic": det,
        "GO": all(c_["passed"] for c_ in checks) and det,
    }


# ─── rung bench (cw): L3 fused vs L2 epilogue-fused vs lever-B batched ────────


def _run_bench(nt: int, b: int, h: int, trials: int) -> dict[str, Any]:
    """Event-path K#1-only ladder timing on one in-process cw bundle.

    All three candidates consume identical inputs; correctness is cross-checked
    against the bundle expected (scale_rel) so a fast-but-wrong rung cannot pass
    silently. Compile walls (L3 bakes nt) land in warmup, outside the timed region.
    """
    from flash_mamba_rl.kernels.cute.gdn2_bwd_dhu_cw import (
        run_k1_incB_batched,
    )
    from flash_mamba_rl.kernels.cute.gdn2_bwd_dhu_l2 import run_k1_incB_l2
    from flash_mamba_rl.verifier.timing import benchmark

    payload = _build_bundle("cw", nt=nt, b=b, h=h)
    inp = {k: v.cuda() for k, v in payload["inputs"].items()}
    exp = payload["expected"]
    k1_args = (
        inp["q"], inp["k"], inp["wy"], inp["g2"], inp["g_last"],
        inp["do"], inp["dv_local"], inp["dht"],
    )

    # benchmark() picks CUDA-event timing from the CALL inputs; arg-less callables
    # fall to the unsynced wall path and read ~0 for async work (the burst-2 v2
    # artifact). Thread a CUDA probe tensor through every timed call.
    sync_probe = torch.zeros(1, device="cuda")

    def bench_cuda(fn: Any) -> float:
        return benchmark(lambda _p: fn(), (sync_probe,), warmup=5, trials=trials).median_ms

    candidates: dict[str, Any] = {
        "l3_fused": lambda: run_k1_incB2_v3(*k1_args, cw=True),
        "l2_epilogue": lambda: run_k1_incB_l2(*k1_args),
        "leverb_batched": lambda: run_k1_incB_batched(*k1_args),
    }

    row: dict[str, Any] = {
        "shape": {"B": b, "H": h, "NT": nt, "L": nt * 64, "c": 64, "d_k": 128, "d_v": 64},
        "trials": trials,
    }
    for name, fn in candidates.items():
        try:
            got = fn()  # warm/JIT + correctness cross-check
            worst = max(
                _scale_rel(g, exp[n]) for g, n in zip(got, ("dh", "dv2", "dh0"), strict=True)
            )
            row[f"{name}_scale_rel"] = worst
            row[f"{name}_ms"] = bench_cuda(fn)
        except Exception:
            row[f"{name}_err"] = traceback.format_exc()
        print(f"  {name}: ms={row.get(f'{name}_ms')} scale_rel={row.get(f'{name}_scale_rel')}",
              flush=True)

    for base in ("l2_epilogue", "leverb_batched"):
        if f"{base}_ms" in row and "l3_fused_ms" in row and row["l3_fused_ms"] > 0:
            row[f"{base}_over_l3"] = row[f"{base}_ms"] / row["l3_fused_ms"]
    return row


# ─── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    # Parse before cutlass import-time argparse fires (tmem_offset_probe.py pattern).
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--desk", action="store_true")
    ap.add_argument("--bench", action="store_true", help="rung bench (cw): L3 vs L2 vs lever-B")
    ap.add_argument("--mode", choices=["scalar", "cw"], default="scalar")
    ap.add_argument("--bundle", type=str, default="")
    ap.add_argument("--nt", type=int, default=4, choices=[1, 2, 4, 8, 16, 32],
                    help="chunks in the in-process bundle (unroll/compile-time stage knob)")
    ap.add_argument("--bh", type=str, default="2,2", help="bench bundle batch,heads (--bench)")
    ap.add_argument("--trials", type=int, default=30, help="timed trials (--bench)")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("-h", "--help", action="help")
    args = ap.parse_args()

    res: dict[str, Any] = {"mode": args.mode, "nt": args.nt}
    try:
        if args.desk:
            res["GO"] = _desk_gate()
        elif args.bench:
            if not torch.cuda.is_available():
                raise RuntimeError("--bench needs CUDA (box)")
            b, h = (int(x) for x in args.bh.split(","))
            res.update(_run_bench(args.nt, b, h, args.trials))
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("CuTe DSL toolchain unavailable (not an sm_100 box)")
            res.update(_run_box(args.bundle, args.mode, args.nt))
    except Exception as exc:
        res["error"] = f"{type(exc).__name__}: {exc}"
        res["trace"] = traceback.format_exc()
        res["GO"] = False

    if args.out:
        dest = Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str))
    print(f"\nGO={res.get('GO')}  mode={args.mode}")


if __name__ == "__main__":
    main()
