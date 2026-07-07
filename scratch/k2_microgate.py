"""On-box K#2 micro-gate: load a bundle, run the WY-VJP kernel, compare dk2/dv/db/dg2.

Loads the .pt from ``scratch/gen_k2_bundle.py`` and runs ``scratch/gdn2_bwd_wy``'s
``run_k2`` against the reference's expected outputs within bf16 tolerances. If the
kernel module is not yet authored, reports that cleanly (the bundle still validates).
Emits ``results/k2_microgate.json``.

    PYTHONPATH=src uv run --no-sync python scratch/k2_microgate.py --bundle k2_bundle_nt4.pt
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import torch


def _compare(name: str, got: torch.Tensor, exp: torch.Tensor, atol: float, rtol: float) -> dict:
    got_f, exp_f = got.float().cpu(), exp.float().cpu()
    diff = (got_f - exp_f).abs()
    finite = bool(torch.isfinite(got_f).all().item())
    ok = bool((diff <= atol + rtol * exp_f.abs()).all().item()) and finite
    return {
        "name": name,
        "shape": list(got_f.shape),
        "max_abs": diff.max().item(),
        "max_rel": (diff / exp_f.abs().clamp_min(1e-6)).max().item(),
        "finite": finite,
        "passed": ok,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=str, required=True)
    ap.add_argument("--atol", type=float, default=2e-2)
    ap.add_argument("--rtol", type=float, default=2e-2)
    ap.add_argument("--out", type=str, default="results/k2_microgate.json")
    args = ap.parse_args()

    out: dict[str, Any] = {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "bundle": args.bundle,
    }
    try:
        payload = torch.load(args.bundle, weights_only=False)
        out["meta"] = payload["meta"]
        try:
            from lethe.kernels.cute.gdn2_bwd_wy import is_available, run_k2
        except ImportError:
            out["kernel_available"] = False
            out["note"] = "scratch/gdn2_bwd_wy not authored yet; bundle validated only"
            out["bundle_finite"] = all(
                bool(torch.isfinite(t_).all().item())
                for t_ in (*payload["inputs"].values(), *payload["expected"].values())
            )
            out["GO"] = False
            raise SystemExit(0)

        out["kernel_available"] = is_available()
        inp = {k: v.cuda() for k, v in payload["inputs"].items()}
        exp = payload["expected"]
        dk2, dv, db, dg2 = run_k2(
            inp["k"], inp["v"], inp["beta"], inp["g2"], inp["T"], inp["dw"], inp["du"]
        )
        checks = [
            _compare("dk2", dk2, exp["dk2"], args.atol, args.rtol),
            _compare("dv", dv, exp["dv"], args.atol, args.rtol),
            _compare("db", db, exp["db"], args.atol, args.rtol),
            _compare("dg2", dg2, exp["dg2"], args.atol, args.rtol),
        ]
        out["checks"] = checks
        out["GO"] = all(c["passed"] for c in checks)
    except SystemExit:
        pass
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
