"""Level-2 de-glue box GATE harness — kernels live in src (promoted at burst-3).

The two epilogue-fused tcgen05 kernels + launcher + pure-torch model were promoted to
``flash_mamba_rl.kernels.cute.gdn2_bwd_dhu_l2`` after the burst-2 silicon GO
(results/k1_l2_epilogue_box.json). This file is the reproducibility layer only:
``--desk-check`` (off-box fp64 model gate) and the box CLI grading
``run_k1_incB_l2`` against the cw K#1 bundle (+ determinism).
"""

import argparse
import json
import traceback
from pathlib import Path

import torch
from torch import Tensor

from flash_mamba_rl.kernels.cute.gdn2_bwd_dhu_l2 import (
    _modelled_l2,
    run_k1_incB_l2,
)

# ------------------------------------------------------------------
# Desk gate: _modelled_l2 vs fp64 build_microgate_bundles_cw expected.
# ------------------------------------------------------------------


def _rel(got: Tensor, ref: Tensor) -> float:
    return (
        (got.float() - ref.float()).abs().max() / ref.float().abs().max().clamp_min(1e-12)
    ).item()


def _l2(x: Tensor) -> Tensor:
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def desk_check() -> bool:
    from flash_mamba_rl.kernels.references.gdn2_chunkwise_cw import build_microgate_bundles_cw

    # All d_v=64 (the kernel tile width). nt ∈ {1,2,3} — nt=1 masked blocker 1 before fix.
    shapes = [
        (1, 1, 1, 64, 128, 64),
        (2, 2, 2, 64, 128, 64),
        (1, 1, 3, 64, 128, 64),
    ]
    worst = 0.0
    for shape in shapes:
        b, h, nt, c, d_k, d_v = shape
        t = nt * c
        gen = torch.Generator().manual_seed(nt * 17 + c)
        dt = torch.float64
        q = _l2(torch.randn(b, t, h, d_k, generator=gen, dtype=dt))
        k = _l2(torch.randn(b, t, h, d_k, generator=gen, dtype=dt))
        v = torch.randn(b, t, h, d_v, generator=gen, dtype=dt)
        g = -(torch.rand(b, t, h, d_k, generator=gen, dtype=dt) * 0.1 + 0.01)
        bg = torch.rand(b, t, h, d_k, generator=gen, dtype=dt) * 0.8 + 0.1
        wg = torch.rand(b, t, h, d_v, generator=gen, dtype=dt) * 0.8 + 0.1
        do = torch.randn(b, t, h, d_v, generator=gen, dtype=dt)
        bun = build_microgate_bundles_cw(q, k, v, g, bg, wg, do, chunk_len=c, scale=d_k**-0.5)
        i1, e1 = bun["k1"].inputs, bun["k1"].expected
        got = _modelled_l2(
            i1["q"],
            i1["k"],
            i1["wy"],
            i1["g2"],
            i1["g_last"],
            i1["do"],
            i1["dv_local"],
            i1["dht"],
        )
        for got_t, name in zip(got, ("dh", "dv2", "dh0"), strict=True):
            rel = _rel(got_t, e1[name])
            worst = max(worst, rel)
        print(f"  shape={shape}  worst_scale_rel_so_far={worst:.2e}")

    tol = 5e-3
    ok = worst < tol
    print(f"\n_modelled_l2 vs fp64 ref: worst_scale_rel={worst:.2e}  tol={tol:.0e}  GO={ok}")
    return ok


def _build_cw_bundle(b: int, h: int, nt: int, c: int, d_k: int, d_v: int) -> dict:
    from flash_mamba_rl.kernels.references.gdn2_chunkwise_cw import build_microgate_bundles_cw

    t = nt * c
    gen = torch.Generator().manual_seed(42)
    dt = torch.float64
    q = _l2(torch.randn(b, t, h, d_k, generator=gen, dtype=dt))
    k = _l2(torch.randn(b, t, h, d_k, generator=gen, dtype=dt))
    v = torch.randn(b, t, h, d_v, generator=gen, dtype=dt)
    g = -(torch.rand(b, t, h, d_k, generator=gen, dtype=dt) * 0.1 + 0.01)
    bg = torch.rand(b, t, h, d_k, generator=gen, dtype=dt) * 0.8 + 0.1
    wg = torch.rand(b, t, h, d_v, generator=gen, dtype=dt) * 0.8 + 0.1
    do = torch.randn(b, t, h, d_v, generator=gen, dtype=dt)
    bun = build_microgate_bundles_cw(q, k, v, g, bg, wg, do, chunk_len=c, scale=d_k**-0.5)
    k1 = bun["k1"]
    return {
        "inputs": {n: t_.float() for n, t_ in k1.inputs.items()},
        "expected": {n: t_.float() for n, t_ in k1.expected.items()},
        "meta": {**k1.meta, "B": b, "H": h, "NT": nt, "C": c, "d_k": d_k, "d_v": d_v},
    }


# ------------------------------------------------------------------
# Box CLI: --mode cw grades run_k1_incB_l2 vs cw bundle.
# ------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--desk-check", action="store_true", help="off-box pure-torch model gate")
    ap.add_argument("--mode", type=str, default="cw", choices=["cw"])
    ap.add_argument(
        "--bundle",
        type=str,
        default="k1_bundle_cw_nt4.pt",
        help="cw K#1 bundle .pt (inputs: q,k,wy,g2,g_last,do,dv_local,dht); "
        "built in-process at (b=2,h=2,nt=4,c=64,d_k=128,d_v=64) if absent",
    )
    ap.add_argument("--atol", type=float, default=5e-3)
    ap.add_argument("--rtol", type=float, default=5e-3)
    ap.add_argument("--out", type=str, default="results/k1_l2_epilogue.json")
    args = ap.parse_args()

    if args.desk_check:
        ok = desk_check()
        raise SystemExit(0 if ok else 1)

    bundle_path = Path(args.bundle)
    if not bundle_path.exists():
        print(f"{bundle_path} not found — building in-process (b=2,h=2,nt=4,c=64,d_k=128,d_v=64)")
        payload = _build_cw_bundle(b=2, h=2, nt=4, c=64, d_k=128, d_v=64)
        torch.save(payload, bundle_path)
        print(f"saved {bundle_path}")
    else:
        payload = torch.load(bundle_path, weights_only=False)

    out: dict = {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "mode": args.mode,
        "bundle": str(bundle_path),
    }
    try:
        inp = {kk: vv.cuda() for kk, vv in payload["inputs"].items()}
        exp = payload["expected"]
        out["meta"] = payload.get("meta", {})

        dh1, dv2_1, dh0_1 = run_k1_incB_l2(
            inp["q"],
            inp["k"],
            inp["wy"],
            inp["g2"],
            inp["g_last"],
            inp["do"],
            inp["dv_local"],
            inp["dht"],
        )
        dh2, dv2_2, dh0_2 = run_k1_incB_l2(
            inp["q"],
            inp["k"],
            inp["wy"],
            inp["g2"],
            inp["g_last"],
            inp["do"],
            inp["dv_local"],
            inp["dht"],
        )

        def _cmp(name: str, got: Tensor, ref: Tensor) -> dict:
            g = got.float().cpu()
            r = ref.float().cpu() if isinstance(ref, Tensor) else torch.tensor(ref)
            diff = (g - r).abs()
            denom = r.abs().clamp_min(1e-6)
            ok_ = bool((diff <= args.atol + args.rtol * r.abs()).all().item())
            return {
                "name": name,
                "max_abs": diff.max().item(),
                "max_rel": (diff / denom).max().item(),
                "finite": bool(torch.isfinite(g).all().item()),
                "passed": ok_ and bool(torch.isfinite(g).all().item()),
            }

        checks = [
            _cmp("dh", dh1, exp["dh"]),
            _cmp("dv2", dv2_1, exp["dv2"]),
            _cmp("dh0", dh0_1, exp["dh0"]),
        ]
        det_ok = (
            torch.equal(dh1.cpu(), dh2.cpu())
            and torch.equal(dv2_1.cpu(), dv2_2.cpu())
            and torch.equal(dh0_1.cpu(), dh0_2.cpu())
        )
        out["checks"] = checks
        out["deterministic"] = det_ok
        out["GO"] = all(c["passed"] for c in checks) and det_ok
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["trace"] = traceback.format_exc()
        out["GO"] = False

    dest = Path(args.out)
    dest.parent.mkdir(exist_ok=True, parents=True)
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))
    print(f"\nGO={out.get('GO')}  ->  {dest}")


if __name__ == "__main__":
    main()
