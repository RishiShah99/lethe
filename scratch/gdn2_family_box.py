"""Family credentials on silicon — GLA / LA / SSD / KDA vs their independent oracles.

Runs each ``native_*_backward`` wrapper (which injects the compiled tcgen05 K#1/K#2)
against the corresponding family oracle (fp64 reference autograd). Grades all grad fields
by scale_rel, checks 2-run bit-determinism, and emits a GO flag.

Dim locks (enforced by the channel-wise kernels): d_k=128, d_v in {64, 128}, L%64==0.
Shapes: (B=2, L=256, H=4, d_k=128, d_v=128) and (B=1, L=512, H=2, d_k=128, d_v=64).
fp32 inputs (our kernels are fp32-safe; oracle is fp64).

KDA runs the FULL path (both K#1 and K#2). No-erase families (GLA, LA, SSD) ride the
skip-T fast path (K#2 not launched; K#1 fed exact-zero wy operand).

Run on box:
  PYTHONPATH=$PWD/src:$PWD ~/cuteenv/bin/python scratch/gdn2_family_box.py \
      --out ~/box_out_burst1/gdn2_family_box.json
"""

import argparse
import json
import traceback
from typing import Any

import torch

ATOL = 5e-3  # integration-gate tolerance; matches gdn2_integration_box_cw.py
SHAPES: list[tuple[int, int, int, int, int]] = [
    (2, 256, 4, 128, 128),
    (1, 512, 2, 128, 64),
]
GRAD_FIELDS_GLA = ("grad_q", "grad_k", "grad_v", "grad_g")
GRAD_FIELDS_LA = ("grad_q", "grad_k", "grad_v")
GRAD_FIELDS_SSD = ("grad_q", "grad_k", "grad_v", "grad_g")
GRAD_FIELDS_KDA = ("grad_q", "grad_k", "grad_v", "grad_g", "grad_beta")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Family silicon credentials")
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def _scale_rel(a: torch.Tensor, e: torch.Tensor) -> float:
    return ((a - e).abs().max() / e.abs().max().clamp_min(1e-12)).item()


def _inputs(
    b: int, ll: int, h: int, dk: int, dv: int, dev: torch.device, seed: int
) -> dict[str, torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def r(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen).to(device=dev, dtype=torch.float32)

    q = r(b, ll, h, dk)
    k = r(b, ll, h, dk)
    v = r(b, ll, h, dv)
    g = -torch.rand(b, ll, h, dk, generator=gen).to(device=dev) * 0.1  # log-decay <= 0
    beta = torch.rand(b, ll, h, generator=gen).to(device=dev) * 0.8 + 0.1  # (0.1, 0.9)
    do = r(b, ll, h, dv)
    return {"q": q, "k": k, "v": v, "g": g, "beta": beta, "do": do}


def _run_family(
    family: str,
    b: int,
    ll: int,
    h: int,
    dk: int,
    dv: int,
    dev: torch.device,
    seed: int,
) -> dict[str, Any]:
    # Lazy import AFTER argparse — avoids cutlass import-time argparse collision.
    from lethe.kernels.cute.gdn2_backward import (
        native_gla_backward,
        native_kda_backward,
        native_la_backward,
        native_ssd_backward,
    )
    from lethe.kernels.references.family_oracles import (
        reference_gla_backward,
        reference_kda_backward,
        reference_la_backward,
        reference_ssd_backward,
    )

    inp = _inputs(b, ll, h, dk, dv, dev, seed)
    q, k, v, g, beta, do = inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"], inp["do"]
    # SSD uses scalar g [B,L,H] — mean over d_k
    g_scalar = g.mean(-1)

    result: dict[str, Any] = {"family": family, "shape": [b, ll, h, dk, dv]}

    try:
        if family == "gla":
            got = native_gla_backward(q, k, v, g, do)
            fields = GRAD_FIELDS_GLA
        elif family == "la":
            got = native_la_backward(q, k, v, do)
            fields = GRAD_FIELDS_LA
        elif family == "ssd":
            got = native_ssd_backward(q, k, v, g_scalar, do)
            fields = GRAD_FIELDS_SSD
        elif family == "kda":
            got = native_kda_backward(q, k, v, g, beta, do)
            fields = GRAD_FIELDS_KDA
        else:
            result["error"] = f"unknown family {family}"
            return result

        if got is None:
            result["dispatch_refused"] = True
            result["note"] = "native_*_backward returned None (is_available/dims check failed)"
            return result

        # fp64 oracle
        q64, k64, v64, g64, beta64, do64 = (x.to(torch.float64) for x in (q, k, v, g, beta, do))
        g_scalar64 = g64.mean(-1)
        if family == "gla":
            orc = reference_gla_backward(q64, k64, v64, g64, do64)
        elif family == "la":
            orc = reference_la_backward(q64, k64, v64, do64)
        elif family == "ssd":
            orc = reference_ssd_backward(q64, k64, v64, g_scalar64, do64)
        elif family == "kda":
            orc = reference_kda_backward(q64, k64, v64, g64, beta64, do64)

        rels: dict[str, float] = {}
        for f in fields:
            cand = getattr(got, f).float()
            ref = getattr(orc, f).float()
            rels[f] = _scale_rel(cand, ref)
        worst = max(rels.values())

        # Determinism: second run
        if family == "gla":
            got2 = native_gla_backward(q, k, v, g, do)
        elif family == "la":
            got2 = native_la_backward(q, k, v, do)
        elif family == "ssd":
            got2 = native_ssd_backward(q, k, v, g_scalar, do)
        else:
            got2 = native_kda_backward(q, k, v, g, beta, do)

        if got2 is None:
            det = False
            det_note = "second run returned None"
        else:
            primary_field = fields[2]  # grad_v is a good determinism sentinel
            det = bool(torch.equal(getattr(got, primary_field), getattr(got2, primary_field)))
            det_note = None

        passed = worst <= ATOL and det
        result.update(
            {
                "scale_rel": rels,
                "worst": worst,
                "deterministic": det,
                "passed": passed,
            }
        )
        if det_note:
            result["det_note"] = det_note
        print(
            f"  {family} shape=({b},{ll},{h},{dk},{dv}) worst={worst:.3e} det={det} "
            f"{'OK' if passed else 'FAIL'}"
        )

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["trace"] = traceback.format_exc()
        result["passed"] = False

    return result


def main() -> None:
    args = _parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}  dtype=fp32 inputs / fp64 oracle")

    families = ["gla", "la", "ssd", "kda"]
    all_results: list[dict[str, Any]] = []
    go = True

    for b, ll, h, dk, dv in SHAPES:
        for fam in families:
            seed = b * 100 + ll + ord(fam[0])
            r = _run_family(fam, b, ll, h, dk, dv, dev, seed)
            all_results.append(r)
            if not r.get("passed", False) and (
                r.get("dispatch_refused") or "error" in r or "scale_rel" in r
            ):
                go = False

    import torch as _torch  # already imported; reuse for env block

    env_block: dict[str, Any] = {
        "torch_version": _torch.__version__,
        "cuda_available": _torch.cuda.is_available(),
        "device_name": _torch.cuda.get_device_name(0) if _torch.cuda.is_available() else None,
    }
    try:
        import cutlass

        env_block["cutlass_dsl_version"] = getattr(cutlass, "__version__", "unknown")
    except ImportError:
        env_block["cutlass_dsl_version"] = "not_importable"

    verdict: dict[str, Any] = {
        "env": env_block,
        "atol": ATOL,
        "shapes": SHAPES,
        "results": all_results,
        "GO": go,
    }
    print(f"\nGO={go}")

    if args.out:
        import pathlib

        dest = pathlib.Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(verdict, indent=2, default=str))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
