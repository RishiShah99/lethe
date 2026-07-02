"""External definitional parity — EXPLORATORY. Not a gate.

Compares our family forward oracles (reference_gla_forward, reference_ssd_forward,
reference_kda_forward) against fla's Triton chunk kernels (chunk_gla, chunk_simple_gla,
chunk_kda) and cuLA's chunk_kda if importable. Small shapes; fp32 (what our refs
accept). Forward-only first; grads only if forward parity is found (<1e-2 scale_rel).

This uses the REFERENCE (pure-torch) forward path — no dim-lock constraints, any shape
works. We do NOT call native_* here.

Convention exploration per family:
- GLA: try fla.ops.gla.chunk_gla with use_qk_l2norm=False/True on our side, scale=d_k**-0.5
  and scale=1.0. fla's g convention: [B,H,L] or [B,L,H,d_k]? Introspect signature, try both.
- SSD: try fla.ops.simple_gla.chunk_simple_gla; our g [B,L,H] → fla scalar per-head g [B,H,L].
- KDA: try fla.ops.kda.chunk_kda (per-channel g, scalar beta). Also try cuLA chunk_kda.
  cuLA import: `from cula.ops.kda import chunk_kda` — record ImportError otherwise.

JSON records: per family per convention combination: scale_rel + verdict.
Top-level: ``definitional_parity`` per family: "established", "not_established", or "skipped".

Run on box:
  PYTHONPATH=$PWD/src:$PWD ~/cuteenv/bin/python scratch/family_external_parity.py \
      --out ~/box_out_burst1/family_external_parity.json
"""

import argparse
import inspect
import json
import traceback
from typing import Any

import torch


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Family external definitional parity (exploratory)")
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


# Small shapes; no dim-lock (refs path).
# (B, L, H, d_k, d_v)
SHAPES: list[tuple[int, int, int, int, int]] = [
    (1, 128, 2, 64, 64),
    (2, 256, 4, 64, 64),
]

ATOL_FWD = 1e-2  # forward parity threshold to decide whether to probe grads


def _scale_rel(a: torch.Tensor, e: torch.Tensor) -> float:
    return ((a - e).abs().max() / e.abs().max().clamp_min(1e-12)).item()


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)


def _make_inputs(
    b: int, ll: int, h: int, dk: int, dv: int, dev: torch.device, seed: int
) -> dict[str, torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def r(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen).to(device=dev, dtype=torch.float32)

    q = r(b, ll, h, dk)
    k = r(b, ll, h, dk)
    v = r(b, ll, h, dv)
    g_cw = (-torch.rand(b, ll, h, dk, generator=gen) * 0.1).to(device=dev)
    g_scalar = g_cw.mean(-1)  # [B, L, H]
    beta = (torch.rand(b, ll, h, generator=gen) * 0.8 + 0.1).to(device=dev)
    return {"q": q, "k": k, "v": v, "g_cw": g_cw, "g_scalar": g_scalar, "beta": beta}


def _probe_gla(dev: torch.device) -> dict[str, Any]:
    """Compare reference_gla_forward vs fla chunk_gla across convention combinations."""
    from flash_mamba_rl.kernels.references.family_oracles import (
        reference_gla_forward,
    )

    result: dict[str, Any] = {"family": "GLA", "fla_op": "fla.ops.gla.chunk_gla"}

    try:
        from fla.ops.gla import chunk_gla

        sig = str(inspect.signature(chunk_gla))
        result["fla_sig"] = sig
    except ImportError as exc:
        result["skipped"] = f"ImportError: {exc}"
        result["definitional_parity"] = "skipped"
        return result

    params = inspect.signature(chunk_gla).parameters

    shape_results: list[dict[str, Any]] = []
    for b, ll, h, dk, dv in SHAPES:
        if dk != dv:
            continue  # chunk_gla may require dk==dv; test square only
        inp = _make_inputs(b, ll, h, dk, dv, dev, seed=b * 100 + ll)
        q, k, v, g_cw = inp["q"], inp["k"], inp["v"], inp["g_cw"]

        conv_results: list[dict[str, Any]] = []
        for use_norm in (True, False):
            for scale_ov in (None, 1.0):
                sc = scale_ov if scale_ov is not None else dk**-0.5
                label = f"norm={use_norm}_scale={'dk^-0.5' if scale_ov is None else scale_ov}"

                # Our reference forward
                try:
                    ours = reference_gla_forward(
                        q.double(),
                        k.double(),
                        v.double(),
                        g_cw.double(),
                        scale=sc,
                        use_qk_l2norm=use_norm,
                    ).float()
                except Exception as exc:
                    conv_results.append({"convention": label, "error": f"oracle:{exc}"})
                    continue

                # fla forward: g convention may be [B,H,L,dk] (transposed) or [B,L,H,dk]
                for g_layout in ("BLHD", "BHLD"):
                    gf = g_cw.transpose(1, 2).contiguous() if g_layout == "BHLD" else g_cw

                    kwargs: dict[str, Any] = {}
                    if "scale" in params:
                        kwargs["scale"] = sc
                    if "use_qk_l2norm" in params:
                        kwargs["use_qk_l2norm"] = use_norm
                    elif "use_qk_l2norm_in_kernel" in params:
                        kwargs["use_qk_l2norm_in_kernel"] = use_norm
                    if "output_final_state" in params:
                        kwargs["output_final_state"] = False

                    try:
                        out_fla = chunk_gla(q, k, v, gf, **kwargs)
                        if isinstance(out_fla, tuple):
                            out_fla = out_fla[0]
                        sr = _scale_rel(out_fla.float(), ours)
                        conv_results.append(
                            {
                                "convention": label,
                                "g_layout": g_layout,
                                "scale_rel_fwd": sr,
                                "passed_fwd": sr < ATOL_FWD,
                            }
                        )
                    except Exception as exc:
                        conv_results.append(
                            {
                                "convention": label,
                                "g_layout": g_layout,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )

        shape_results.append({"shape": [b, ll, h, dk, dv], "conventions": conv_results})

    result["per_shape"] = shape_results
    all_srs = [
        c["scale_rel_fwd"] for s in shape_results for c in s["conventions"] if "scale_rel_fwd" in c
    ]
    best_sr = min(all_srs) if all_srs else None
    result["best_scale_rel_fwd"] = best_sr
    result["definitional_parity"] = (
        "established" if (best_sr is not None and best_sr < ATOL_FWD) else "not_established"
    )
    return result


def _probe_ssd(dev: torch.device) -> dict[str, Any]:
    """Compare reference_ssd_forward vs fla chunk_simple_gla."""
    from flash_mamba_rl.kernels.references.family_oracles import (
        reference_ssd_forward,
    )

    result: dict[str, Any] = {"family": "SSD", "fla_op": "fla.ops.simple_gla.chunk_simple_gla"}

    try:
        from fla.ops.simple_gla import chunk_simple_gla

        result["fla_sig"] = str(inspect.signature(chunk_simple_gla))
    except ImportError as exc:
        result["skipped"] = f"ImportError: {exc}"
        result["definitional_parity"] = "skipped"
        return result

    params = inspect.signature(chunk_simple_gla).parameters
    shape_results: list[dict[str, Any]] = []

    for b, ll, h, dk, dv in SHAPES:
        if dk != dv:
            continue
        inp = _make_inputs(b, ll, h, dk, dv, dev, seed=b * 200 + ll)
        q, k, v, g_scalar = inp["q"], inp["k"], inp["v"], inp["g_scalar"]

        conv_results: list[dict[str, Any]] = []
        for use_norm in (True, False):
            for scale_ov in (None, 1.0):
                sc = scale_ov if scale_ov is not None else dk**-0.5
                label = f"norm={use_norm}_scale={'dk^-0.5' if scale_ov is None else scale_ov}"

                try:
                    ours = reference_ssd_forward(
                        q.double(),
                        k.double(),
                        v.double(),
                        g_scalar.double(),
                        scale=sc,
                        use_qk_l2norm=use_norm,
                    ).float()
                except Exception as exc:
                    conv_results.append({"convention": label, "error": f"oracle:{exc}"})
                    continue

                # fla simple_gla g: expect [B,L,H] or [B,H,L]
                for g_layout in ("BLH", "BHL"):
                    gf = g_scalar.transpose(1, 2).contiguous() if g_layout == "BHL" else g_scalar

                    kwargs: dict[str, Any] = {}
                    if "scale" in params:
                        kwargs["scale"] = sc
                    if "use_qk_l2norm" in params:
                        kwargs["use_qk_l2norm"] = use_norm
                    if "output_final_state" in params:
                        kwargs["output_final_state"] = False

                    try:
                        out_fla = chunk_simple_gla(q, k, v, gf, **kwargs)
                        if isinstance(out_fla, tuple):
                            out_fla = out_fla[0]
                        sr = _scale_rel(out_fla.float(), ours)
                        conv_results.append(
                            {
                                "convention": label,
                                "g_layout": g_layout,
                                "scale_rel_fwd": sr,
                                "passed_fwd": sr < ATOL_FWD,
                            }
                        )
                    except Exception as exc:
                        conv_results.append(
                            {
                                "convention": label,
                                "g_layout": g_layout,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )

        shape_results.append({"shape": [b, ll, h, dk, dv], "conventions": conv_results})

    result["per_shape"] = shape_results
    all_srs = [
        c["scale_rel_fwd"] for s in shape_results for c in s["conventions"] if "scale_rel_fwd" in c
    ]
    best_sr = min(all_srs) if all_srs else None
    result["best_scale_rel_fwd"] = best_sr
    result["definitional_parity"] = (
        "established" if (best_sr is not None and best_sr < ATOL_FWD) else "not_established"
    )
    return result


def _probe_kda_fla(dev: torch.device) -> dict[str, Any]:
    """Compare reference_kda_forward vs fla chunk_kda."""
    from flash_mamba_rl.kernels.references.family_oracles import (
        reference_kda_forward,
    )

    result: dict[str, Any] = {"family": "KDA_fla", "fla_op": "fla.ops.kda.chunk_kda"}

    try:
        from fla.ops.kda import chunk_kda

        result["fla_sig"] = str(inspect.signature(chunk_kda))
    except ImportError as exc:
        result["skipped"] = f"ImportError: {exc}"
        result["definitional_parity"] = "skipped"
        return result

    params = inspect.signature(chunk_kda).parameters
    shape_results: list[dict[str, Any]] = []

    for b, ll, h, dk, dv in SHAPES:
        if dk != dv:
            continue
        inp = _make_inputs(b, ll, h, dk, dv, dev, seed=b * 300 + ll)
        q, k, v, g_cw, beta = inp["q"], inp["k"], inp["v"], inp["g_cw"], inp["beta"]

        conv_results: list[dict[str, Any]] = []
        for use_norm in (True, False):
            for scale_ov in (None, 1.0):
                sc = scale_ov if scale_ov is not None else dk**-0.5
                label = f"norm={use_norm}_scale={'dk^-0.5' if scale_ov is None else scale_ov}"

                try:
                    ours = reference_kda_forward(
                        q.double(),
                        k.double(),
                        v.double(),
                        g_cw.double(),
                        beta.double(),
                        scale=sc,
                        use_qk_l2norm=use_norm,
                    ).float()
                except Exception as exc:
                    conv_results.append({"convention": label, "error": f"oracle:{exc}"})
                    continue

                for g_layout in ("BLHD", "BHLD"):
                    gf = g_cw.transpose(1, 2).contiguous() if g_layout == "BHLD" else g_cw

                    # beta: [B,L,H] or [B,H,L]
                    for beta_layout in ("BLH", "BHL"):
                        betaf = beta.transpose(1, 2).contiguous() if beta_layout == "BHL" else beta

                        kwargs: dict[str, Any] = {}
                        if "scale" in params:
                            kwargs["scale"] = sc
                        if "use_qk_l2norm" in params:
                            kwargs["use_qk_l2norm"] = use_norm
                        if "output_final_state" in params:
                            kwargs["output_final_state"] = False

                        try:
                            out_fla = chunk_kda(q, k, v, gf, betaf, **kwargs)
                            if isinstance(out_fla, tuple):
                                out_fla = out_fla[0]
                            sr = _scale_rel(out_fla.float(), ours)
                            conv_results.append(
                                {
                                    "convention": label,
                                    "g_layout": g_layout,
                                    "beta_layout": beta_layout,
                                    "scale_rel_fwd": sr,
                                    "passed_fwd": sr < ATOL_FWD,
                                }
                            )
                        except Exception as exc:
                            conv_results.append(
                                {
                                    "convention": label,
                                    "g_layout": g_layout,
                                    "beta_layout": beta_layout,
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                            )

        shape_results.append({"shape": [b, ll, h, dk, dv], "conventions": conv_results})

    result["per_shape"] = shape_results
    all_srs = [
        c["scale_rel_fwd"] for s in shape_results for c in s["conventions"] if "scale_rel_fwd" in c
    ]
    best_sr = min(all_srs) if all_srs else None
    result["best_scale_rel_fwd"] = best_sr
    result["definitional_parity"] = (
        "established" if (best_sr is not None and best_sr < ATOL_FWD) else "not_established"
    )
    return result


def _probe_kda_cula(dev: torch.device) -> dict[str, Any]:
    """Compare reference_kda_forward vs cuLA chunk_kda (if importable)."""
    from flash_mamba_rl.kernels.references.family_oracles import (
        reference_kda_forward,
    )

    result: dict[str, Any] = {"family": "KDA_cuLA", "cula_op": "cula.ops.kda.chunk_kda"}

    try:
        from cula.ops.kda import chunk_kda  # type: ignore[import]

        result["cula_sig"] = str(inspect.signature(chunk_kda))
    except ImportError as exc:
        result["skipped"] = f"ImportError: {exc}"
        result["definitional_parity"] = "skipped"
        return result
    except Exception as exc:
        result["skipped"] = f"{type(exc).__name__}: {exc}"
        result["definitional_parity"] = "skipped"
        return result

    params = inspect.signature(chunk_kda).parameters
    shape_results: list[dict[str, Any]] = []

    for b, ll, h, dk, dv in SHAPES:
        if dk != dv:
            continue
        inp = _make_inputs(b, ll, h, dk, dv, dev, seed=b * 400 + ll)
        q, k, v, g_cw, beta = inp["q"], inp["k"], inp["v"], inp["g_cw"], inp["beta"]

        conv_results: list[dict[str, Any]] = []
        for use_norm in (True, False):
            for scale_ov in (None, 1.0):
                sc = scale_ov if scale_ov is not None else dk**-0.5
                label = f"norm={use_norm}_scale={'dk^-0.5' if scale_ov is None else scale_ov}"

                try:
                    ours = reference_kda_forward(
                        q.double(),
                        k.double(),
                        v.double(),
                        g_cw.double(),
                        beta.double(),
                        scale=sc,
                        use_qk_l2norm=use_norm,
                    ).float()
                except Exception as exc:
                    conv_results.append({"convention": label, "error": f"oracle:{exc}"})
                    continue

                kwargs: dict[str, Any] = {}
                if "scale" in params:
                    kwargs["scale"] = sc
                if "use_qk_l2norm" in params:
                    kwargs["use_qk_l2norm"] = use_norm

                try:
                    out_cula = chunk_kda(q, k, v, g_cw, beta, **kwargs)
                    if isinstance(out_cula, tuple):
                        out_cula = out_cula[0]
                    sr = _scale_rel(out_cula.float(), ours)
                    conv_results.append(
                        {
                            "convention": label,
                            "scale_rel_fwd": sr,
                            "passed_fwd": sr < ATOL_FWD,
                        }
                    )
                except Exception as exc:
                    conv_results.append(
                        {
                            "convention": label,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

        shape_results.append({"shape": [b, ll, h, dk, dv], "conventions": conv_results})

    result["per_shape"] = shape_results
    all_srs = [
        c["scale_rel_fwd"] for s in shape_results for c in s["conventions"] if "scale_rel_fwd" in c
    ]
    best_sr = min(all_srs) if all_srs else None
    result["best_scale_rel_fwd"] = best_sr
    result["definitional_parity"] = (
        "established" if (best_sr is not None and best_sr < ATOL_FWD) else "not_established"
    )
    return result


def main() -> None:
    args = _parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}  note=exploratory forward parity only")

    probes = [
        ("GLA", _probe_gla),
        ("SSD", _probe_ssd),
        ("KDA_fla", _probe_kda_fla),
        ("KDA_cuLA", _probe_kda_cula),
    ]

    results: dict[str, Any] = {}
    for name, fn in probes:
        print(f"\n--- {name} ---")
        try:
            r = fn(dev)
        except Exception as exc:
            r = {
                "family": name,
                "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc(),
                "definitional_parity": "error",
            }
        results[name] = r
        print(
            f"  definitional_parity={r.get('definitional_parity')}  "
            f"best_sr={r.get('best_scale_rel_fwd')}"
        )

    env_block: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    out: dict[str, Any] = {
        "env": env_block,
        "shapes": SHAPES,
        "atol_fwd": ATOL_FWD,
        "note": "exploratory; grads only attempted if forward parity established (<1e-2)",
        "families": results,
    }

    if args.out:
        import pathlib

        dest = pathlib.Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=2, default=str))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
