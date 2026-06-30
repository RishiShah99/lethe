"""On-box K#1 micro-gate: load a bundle, run the CuTe kernel, compare dh/dh0/dv2.

Loads the .pt written by ``scratch/gen_k1_bundle.py``, runs ``scratch/gdn2_bwd_dhu``'s
``run_k1`` on the B200, and checks the three outputs against the reference's expected
values within the verifier's bf16 tolerances. Emits ``results/k1_microgate.json``.

Run on the box:
    PYTHONPATH=src uv run --no-sync python scratch/k1_microgate.py --bundle k1_bundle_nt1.pt
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import torch


def _compare(name: str, got: torch.Tensor, exp: torch.Tensor, atol: float, rtol: float) -> dict:
    got_f = got.float().cpu()
    exp_f = exp.float().cpu()
    diff = (got_f - exp_f).abs()
    denom = exp_f.abs().clamp_min(1e-6)
    max_abs = diff.max().item()
    max_rel = (diff / denom).max().item()
    ok = bool((diff <= atol + rtol * exp_f.abs()).all().item())
    return {
        "name": name,
        "shape": list(got_f.shape),
        "max_abs": max_abs,
        "max_rel": max_rel,
        "finite": bool(torch.isfinite(got_f).all().item()),
        "passed": ok and bool(torch.isfinite(got_f).all().item()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=str, required=True)
    ap.add_argument("--mode", type=str, default="incA",
                    choices=["incA", "incB_host", "incB", "incB2"])
    ap.add_argument("--atol", type=float, default=2e-2)  # bf16 floor (fla test_gdn.py-class)
    ap.add_argument("--rtol", type=float, default=2e-2)
    ap.add_argument("--out", type=str, default="results/k1_microgate.json")
    args = ap.parse_args()

    out: dict[str, Any] = {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "bundle": args.bundle,
        "mode": args.mode,
    }
    try:
        from scratch import gdn2_bwd_dhu as kmod  # noqa: PLC0415

        runner = {
            "incA": kmod.run_k1,
            "incB_host": kmod.run_k1_incB_host,
            "incB": getattr(kmod, "run_k1_incB", kmod.run_k1_incB_host),
            "incB2": kmod.run_k1_incB2,
        }[args.mode]
        out["kernel_available"] = kmod.is_available()
        payload = torch.load(args.bundle, weights_only=False)
        inp = {k: v.cuda() for k, v in payload["inputs"].items()}
        exp = payload["expected"]
        out["meta"] = payload["meta"]

        dh, dv2, dh0 = runner(
            inp["q"], inp["k"], inp["w"], inp["g2"], inp["g_last"],
            inp["do"], inp["dv_local"], inp["dht"],
        )
        checks = [
            _compare("dh", dh, exp["dh"], args.atol, args.rtol),
            _compare("dv2", dv2, exp["dv2"], args.atol, args.rtol),
            _compare("dh0", dh0, exp["dh0"], args.atol, args.rtol),
        ]
        out["checks"] = checks
        out["GO"] = all(c["passed"] for c in checks)
    except Exception as exc:  # noqa: BLE001
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
