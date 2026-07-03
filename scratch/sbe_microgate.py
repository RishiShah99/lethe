"""Stage-B einsum kernel box GATE + RUNG-BENCH — kernel lives in src (gdn2_sb_einsum).

(default)     box gate: run_sb_einsum vs the fp64 torch einsums over masked_decay_rel
              at --nt/--bh/--gscale (GO = both outputs scale_rel < 5e-3 + 2-run
              bit-determinism)
--drift       the training-drifted regime (--gscale 40: within-chunk log2 span > 128)
--bench       event-path rung timing: the kernel (1 launch, no decay_rel) vs the torch
              path (masked_decay_rel build + 2 einsums), correctness cross-checked.

  PYTHONPATH=src ~/cuteenv/bin/python scratch/sbe_microgate.py --nt 4 --out results/sbe_microgate_nt4.json
"""

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def _build_inputs(nt: int, b: int, h: int, gscale: float, seed: int = 47) -> dict[str, Tensor]:
    """Stage-B-shaped fp64 inputs: lower-incl coef + within-chunk cumsum g2."""
    c, d_k = 64, 128
    gen = torch.Generator().manual_seed(seed)
    dt = torch.float64
    z = (b, h, nt)
    coef = torch.randn(*z, c, c, generator=gen, dtype=dt)
    lower_incl = torch.tril(torch.ones(c, c, dtype=torch.bool), 0)
    coef = torch.where(lower_incl, coef, torch.zeros_like(coef))
    k = torch.randn(*z, c, d_k, generator=gen, dtype=dt)
    q = torch.randn(*z, c, d_k, generator=gen, dtype=dt)
    g_tok = -(torch.rand(*z, c, d_k, generator=gen, dtype=dt) * 0.1 + 0.01) * gscale
    g2 = torch.cumsum(g_tok, dim=-2) * (1.0 / torch.log(torch.tensor(2.0, dtype=dt)))
    return {"coef": coef, "k": k, "q": q, "g2": g2}


def _expected(inp: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
    from flash_mamba_rl.kernels.references.gdn2_chunkwise_cw import masked_decay_rel

    decay_rel = masked_decay_rel(inp["g2"])
    dq_intra = torch.einsum("...is,...sd,...isd->...id", inp["coef"], inp["k"], decay_rel)
    dk_intra = torch.einsum("...is,...id,...isd->...sd", inp["coef"], inp["q"], decay_rel)
    return dq_intra, dk_intra


def _scale_rel(got: Tensor, exp: Tensor) -> float:
    diff = (got.float().cpu() - exp.float().cpu()).abs()
    return (diff.max() / exp.float().cpu().abs().max().clamp(min=1e-12)).item()


def _run_box(nt: int, b: int, h: int, gscale: float) -> dict[str, Any]:
    from flash_mamba_rl.kernels.cute.gdn2_sb_einsum import run_sb_einsum

    inp = _build_inputs(nt, b, h, gscale)
    exp_dqi, exp_dki = _expected(inp)
    cu = {k_: v_.cuda() for k_, v_ in inp.items()}

    got = run_sb_einsum(cu["coef"], cu["k"], cu["q"], cu["g2"])
    checks = []
    for name, g_, e_ in (("dq_intra", got[0], exp_dqi), ("dk_intra", got[1], exp_dki)):
        sr = _scale_rel(g_, e_)
        finite = bool(torch.isfinite(g_).all().item())
        checks.append(
            {"name": name, "scale_rel": sr, "finite": finite, "passed": finite and sr < 5e-3}
        )

    got2 = run_sb_einsum(cu["coef"], cu["k"], cu["q"], cu["g2"])
    det = all(torch.equal(a.cpu(), b_.cpu()) for a, b_ in zip(got, got2, strict=True))
    return {
        "device": torch.cuda.get_device_name(0),
        "shape": {"B": b, "H": h, "NT": nt, "C": 64, "d_k": 128, "gscale": gscale},
        "checks": checks,
        "deterministic": det,
        "GO": all(c_["passed"] for c_ in checks) and det,
    }


def _run_bench(nt: int, b: int, h: int, trials: int) -> dict[str, Any]:
    from flash_mamba_rl.kernels.cute.gdn2_sb_einsum import run_sb_einsum
    from flash_mamba_rl.kernels.references.gdn2_chunkwise_cw import masked_decay_rel
    from flash_mamba_rl.verifier.timing import benchmark

    inp = _build_inputs(nt, b, h, 1.0)
    exp_dqi, exp_dki = _expected(inp)
    cu = {k_: v_.cuda().float() for k_, v_ in inp.items()}
    sync_probe = torch.zeros(1, device="cuda")

    def bench_cuda(fn: Any) -> float:
        return benchmark(lambda _p: fn(), (sync_probe,), warmup=5, trials=trials).median_ms

    def torch_path() -> tuple[Tensor, Tensor]:
        decay_rel = masked_decay_rel(cu["g2"])
        dqi = torch.einsum("...is,...sd,...isd->...id", cu["coef"], cu["k"], decay_rel)
        dki = torch.einsum("...is,...id,...isd->...sd", cu["coef"], cu["q"], decay_rel)
        return dqi, dki

    row: dict[str, Any] = {"shape": {"B": b, "H": h, "NT": nt}, "trials": trials}
    for name, fn in (
        ("sbe_kernel", lambda: run_sb_einsum(*(cu[x] for x in ("coef", "k", "q", "g2")))),
        ("torch_einsums", torch_path),
    ):
        try:
            got = fn()
            worst = max(_scale_rel(g_, e_) for g_, e_ in zip(got, (exp_dqi, exp_dki), strict=True))
            row[f"{name}_scale_rel"] = worst
            row[f"{name}_ms"] = bench_cuda(fn)
        except Exception:
            row[f"{name}_err"] = traceback.format_exc()
        print(
            f"  {name}: ms={row.get(f'{name}_ms')} scale_rel={row.get(f'{name}_scale_rel')}",
            flush=True,
        )
    if row.get("sbe_kernel_ms"):
        row["torch_over_sbe"] = row["torch_einsums_ms"] / row["sbe_kernel_ms"]
    return row


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--drift", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--nt", type=int, default=4)
    ap.add_argument("--bh", type=str, default="2,2")
    ap.add_argument("--gscale", type=float, default=1.0)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("-h", "--help", action="help")
    args = ap.parse_args()

    b, h = (int(x) for x in args.bh.split(","))
    res: dict[str, Any] = {"nt": args.nt, "B": b, "H": h}
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("needs CUDA (sm_100 box)")
        if args.bench:
            res.update(_run_bench(args.nt, b, h, args.trials))
        else:
            res.update(_run_box(args.nt, b, h, 40.0 if args.drift else args.gscale))
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
