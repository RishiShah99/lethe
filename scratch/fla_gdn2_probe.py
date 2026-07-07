"""fla GDN-2 Triton backward probe — the real speed bar vs our native tcgen05 path.

Introspects fla for its GDN-2 backward op, maps our parameterization onto it across
four convention combinations, grades the six GDN-2 grads vs the fp64 oracle, times the
backward, and checks 2-run determinism. Records everything to JSON whether or not any
mapping works — "not_established" is a valid result.

fla GDN-2 parameterization notes (recorded here so the JSON needs no prose annotation):
- fla uses ``chunk_gated_delta_rule_2`` / ``gated_deltanet_2`` as the cw-backward op name;
  the scalar predecessor is ``chunk_gated_delta_rule``.
- Our recurrence:  v_new = (w x v) - (b x k)^T @ S;  b/w are per-channel gates in (0,1).
- fla's GDN-2 call signature (as of 0.5.x): ``chunk_gated_delta_rule_2(q, k, v, g, b, w,
  scale, use_qk_l2norm_in_kernel)`` where b/w match our b/w shapes — but fla may apply
  sigmoid internally; we provide already-sigmoided gates (values already in (0,1)).
- We try use_qk_l2norm True AND False combined with scale=d_k**-0.5 and scale=1.0.
- fp64 oracle grads are reference_gdn2_backward; we grade fla's returned grads by the
  six field names (grad_q, grad_k, grad_v, grad_g, grad_b, grad_w).

Run on box:
  PYTHONPATH=$PWD/src:$PWD ~/cuteenv/bin/python scratch/fla_gdn2_probe.py \
      --out ~/box_out_burst1/fla_gdn2_probe.json
"""

import argparse
import inspect
import json
import traceback
import types
from pathlib import Path
from typing import Any

import torch

# Lazy import of our oracle — avoids cutlass import-time argparse collision.
# (argparse is fully parsed before we touch anything that pulls in cutlass.)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="fla GDN-2 backward probe")
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


# (B, L, H, d_k, d_v) — bf16 inputs; fp64 oracle for grading.
SHAPES: list[tuple[int, int, int, int, int]] = [
    (2, 512, 8, 128, 128),
    (2, 2048, 8, 128, 128),
    (2, 4096, 8, 128, 128),
    (8, 2048, 8, 128, 128),
]

GRAD_FIELDS = ("grad_q", "grad_k", "grad_v", "grad_g", "grad_b", "grad_w")
ATOL = 5e-2  # exploratory; fla is bf16, oracle is fp64

WARMUP = 5
REPS = 20


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)


def _make_inputs(
    b: int, ll: int, h: int, dk: int, dv: int, dtype: torch.dtype, dev: torch.device, seed: int
) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(b, ll, h, dk, generator=gen).to(device=dev, dtype=dtype)
    k = torch.randn(b, ll, h, dk, generator=gen).to(device=dev, dtype=dtype)
    v = torch.randn(b, ll, h, dv, generator=gen).to(device=dev, dtype=dtype)
    g = (-torch.rand(b, ll, h, dk, generator=gen) * 0.1).to(device=dev, dtype=dtype)
    bg = torch.rand(b, ll, h, dk, generator=gen).sigmoid().to(device=dev, dtype=dtype)
    wg = torch.rand(b, ll, h, dv, generator=gen).sigmoid().to(device=dev, dtype=dtype)
    do = torch.randn(b, ll, h, dv, generator=gen).to(device=dev, dtype=dtype)
    return q, k, v, g, bg, wg, do


def _scale_rel(a: torch.Tensor, e: torch.Tensor) -> float:
    return ((a - e).abs().max() / e.abs().max().clamp_min(1e-12)).item()


def _introspect_fla() -> dict[str, Any]:
    """Try to find the fla GDN-2 backward op and record what is actually present."""
    result: dict[str, Any] = {"candidates": [], "found": None, "error": None}
    try:
        import fla
        import fla.ops

        # Collect all submodule names in fla.ops
        op_names = []
        for name in dir(fla.ops):
            attr = getattr(fla.ops, name, None)
            if isinstance(attr, types.ModuleType) or callable(attr):
                op_names.append(name)
        result["fla_ops_attrs"] = op_names[:60]  # truncate for JSON size

        # Try explicit candidate names
        candidates = [
            "gdn2",
            "gated_deltanet_2",
            "gated_delta_rule_2",
            "chunk_gated_delta_rule_2",
            "chunk_gated_deltanet_2",
        ]
        found_ops: dict[str, Any] = {}
        for cname in candidates:
            # Try fla.ops.<cname>
            mod = None
            try:
                mod = getattr(fla.ops, cname, None)
                if mod is None:
                    import importlib

                    mod = importlib.import_module(f"fla.ops.{cname}")
            except (ImportError, AttributeError):
                pass
            if mod is not None:
                sub_callables = {}
                for attr_name in dir(mod):
                    if not attr_name.startswith("_"):
                        attr = getattr(mod, attr_name, None)
                        if callable(attr):
                            try:
                                sig = str(inspect.signature(attr))
                            except (ValueError, TypeError):
                                sig = "<no signature>"
                            sub_callables[attr_name] = sig
                found_ops[cname] = sub_callables
        result["candidates"] = found_ops

        # Pick the most likely backward function
        priority = [
            ("gated_deltanet_2", "chunk_gated_delta_rule_2"),
            ("gdn2", "chunk_gated_delta_rule_2"),
            ("gated_delta_rule_2", "chunk_gated_delta_rule_2"),
        ]
        for mod_name, fn_name in priority:
            try:
                import importlib

                mod = importlib.import_module(f"fla.ops.{mod_name}")
                fn = getattr(mod, fn_name, None)
                if fn is not None:
                    result["found"] = {
                        "module": f"fla.ops.{mod_name}",
                        "fn": fn_name,
                        "sig": str(inspect.signature(fn)),
                    }
                    break
            except (ImportError, AttributeError):
                pass

        # Fallback: check fla.ops.gated_delta_rule for the scalar predecessor
        if result["found"] is None:
            try:
                from fla.ops.gated_delta_rule import chunk_gated_delta_rule

                result["found"] = {
                    "module": "fla.ops.gated_delta_rule",
                    "fn": "chunk_gated_delta_rule",
                    "sig": str(inspect.signature(chunk_gated_delta_rule)),
                    "note": "scalar predecessor only; b/w not separately parameterizable",
                }
            except ImportError:
                pass

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["trace"] = traceback.format_exc()
    return result


def _try_fla_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b_gate: torch.Tensor,
    w_gate: torch.Tensor,
    do: torch.Tensor,
    use_qk_l2norm: bool,
    scale: float,
    found_info: dict[str, Any],
) -> dict[str, Any] | None:
    """Try to call the fla GDN-2 backward for one convention combination.

    Returns a dict with the grad tensors (float64-cast) and metadata, or None
    if the op is not importable or fails with a shape/API mismatch.
    """
    if found_info.get("found") is None:
        return None

    try:
        mod_name = found_info["found"]["module"]
        fn_name = found_info["found"]["fn"]
        import importlib

        mod = importlib.import_module(mod_name)
        fn = getattr(mod, fn_name)

        # Prepare leaf inputs that require grad so autograd can return grads.
        qf = q.float().detach().requires_grad_(True)
        kf = k.float().detach().requires_grad_(True)
        vf = v.float().detach().requires_grad_(True)
        gf = g.float().detach().requires_grad_(True)
        bf = b_gate.float().detach().requires_grad_(True)
        wf = w_gate.float().detach().requires_grad_(True)

        # g convention for fla chunk_gated_delta_rule_2 is per-channel [B,L,H,d_k] log-decay
        # (matches ours); b and w are already-sigmoided gates — fla may expect raw logits.
        # We try two sub-combinations: passing our already-sigmoided b/w directly vs pre-computing
        # g as per-head (collapsing channel axis). Both attempts are guarded per call.

        # Attempt 1: pass b/w as-is (already in (0,1)) with full channel-wise g
        sig = inspect.signature(fn)
        param_names = list(sig.parameters.keys())

        kwargs: dict[str, Any] = {}
        # Detect whether fn takes b/w separately
        has_b = "b" in param_names or "beta" in param_names
        has_w = "w" in param_names

        if "use_qk_l2norm" in param_names or "use_qk_l2norm_in_kernel" in param_names:
            norm_key = (
                "use_qk_l2norm" if "use_qk_l2norm" in param_names else "use_qk_l2norm_in_kernel"
            )
            kwargs[norm_key] = use_qk_l2norm

        if "scale" in param_names:
            kwargs["scale"] = scale

        if "output_final_state" in param_names:
            kwargs["output_final_state"] = False

        # Build positional args based on detected signature
        if has_b and has_w:
            positional = (qf, kf, vf, gf, bf, wf)
        elif "beta" in param_names:
            # scalar beta: b/w must match; use b_gate (key axis) as beta
            positional = (qf, kf, vf, gf, bf)
        else:
            # Scalar-only API: collapse g to per-head (mean over d_k) and use b as scalar beta
            g_head = gf.mean(-1)
            beta_head = bf.mean(-1)
            positional = (qf, kf, vf, g_head, beta_head)

        out = fn(*positional, **kwargs)
        fwd_out = out[0] if isinstance(out, tuple) else out

        dof = do.float()
        grads = torch.autograd.grad(fwd_out, (qf, kf, vf, gf, bf, wf), dof, allow_unused=True)

        return {
            "grad_q": grads[0].double() if grads[0] is not None else None,
            "grad_k": grads[1].double() if grads[1] is not None else None,
            "grad_v": grads[2].double() if grads[2] is not None else None,
            "grad_g": grads[3].double() if grads[3] is not None else None,
            "grad_b": grads[4].double() if grads[4] is not None else None,
            "grad_w": grads[5].double() if grads[5] is not None else None,
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()}


def _time_fla_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b_gate: torch.Tensor,
    w_gate: torch.Tensor,
    do: torch.Tensor,
    found_info: dict[str, Any],
    best_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Time the fla backward using CUDA events (5 warmup, 20 reps, median)."""
    if found_info.get("found") is None:
        return {"error": "no fla op found"}
    try:
        mod_name = found_info["found"]["module"]
        fn_name = found_info["found"]["fn"]
        import importlib

        mod = importlib.import_module(mod_name)
        fn = getattr(mod, fn_name)

        sig = inspect.signature(fn)
        param_names = list(sig.parameters.keys())
        has_b = "b" in param_names or "beta" in param_names
        has_w = "w" in param_names

        def _run() -> None:
            qf = q.float().detach().requires_grad_(True)
            kf = k.float().detach().requires_grad_(True)
            vf = v.float().detach().requires_grad_(True)
            gf = g.float().detach().requires_grad_(True)
            bf = b_gate.float().detach().requires_grad_(True)
            wf = w_gate.float().detach().requires_grad_(True)

            kwargs: dict[str, Any] = {}
            if "output_final_state" in param_names:
                kwargs["output_final_state"] = False
            kwargs.update(best_kwargs)

            if has_b and has_w:
                pos = (qf, kf, vf, gf, bf, wf)
            elif "beta" in param_names:
                pos = (qf, kf, vf, gf, bf)
            else:
                g_head = gf.mean(-1)
                beta_head = bf.mean(-1)
                pos = (qf, kf, vf, g_head, beta_head)

            out = fn(*pos, **kwargs)
            fwd_out = out[0] if isinstance(out, tuple) else out
            dof = do.float()
            torch.autograd.grad(fwd_out, (qf, kf, vf, gf, bf, wf), dof, allow_unused=True)

        # Warmup
        for _ in range(WARMUP):
            _run()
        torch.cuda.synchronize()

        # Timed reps
        times_ms: list[float] = []
        for _ in range(REPS):
            ev_start = torch.cuda.Event(enable_timing=True)
            ev_end = torch.cuda.Event(enable_timing=True)
            ev_start.record()
            _run()
            ev_end.record()
            torch.cuda.synchronize()
            times_ms.append(ev_start.elapsed_time(ev_end))

        times_ms.sort()
        median_ms = times_ms[len(times_ms) // 2]
        return {
            "median_ms": median_ms,
            "min_ms": times_ms[0],
            "max_ms": times_ms[-1],
            "reps": REPS,
            "warmup": WARMUP,
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()}


def main() -> None:
    args = _parse_args()

    try:
        import fla
    except ImportError as exc:
        res = {"error": f"fla not importable: {exc}", "GO": False}
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
        return

    fla_version = getattr(fla, "__version__", "unknown")

    from lethe.kernels.references.gdn_backward import (
        reference_gdn2_backward,
    )

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}")

    introspection = _introspect_fla()
    print(f"fla_version={fla_version}  found={introspection.get('found')}")

    # Convention combinations to try per shape:
    # (use_qk_l2norm, scale_override)  — scale_override=None uses d_k**-0.5
    conventions: list[tuple[bool, float | None]] = [
        (True, None),
        (True, 1.0),
        (False, None),
        (False, 1.0),
    ]

    per_shape: dict[str, Any] = {}
    for b, ll, h, dk, dv in SHAPES:
        shape_key = f"B{b}_L{ll}_H{h}_dk{dk}_dv{dv}"
        print(f"\nshape={shape_key}")
        q, k, v, g, bg, wg, do = _make_inputs(
            b, ll, h, dk, dv, torch.bfloat16, dev, seed=b * 1000 + ll
        )

        # fp64 oracle
        orc = reference_gdn2_backward(*(x.to(torch.float64) for x in (q, k, v, g, bg, wg, do)))
        orc_fields = {f: getattr(orc, f).double() for f in GRAD_FIELDS}

        conv_results: list[dict[str, Any]] = []
        best_scale_rel: dict[str, float] = {}
        best_kwargs: dict[str, Any] = {}

        for use_norm, scale_ov in conventions:
            sc = scale_ov if scale_ov is not None else dk**-0.5
            label = f"norm={use_norm}_scale={'d_k^-0.5' if scale_ov is None else scale_ov}"
            kw: dict[str, Any] = {}
            if scale_ov is None:
                kw["scale"] = dk**-0.5
            else:
                kw["scale"] = scale_ov
            if use_norm:
                kw["use_qk_l2norm"] = True
            else:
                kw["use_qk_l2norm"] = False

            got = _try_fla_backward(q, k, v, g, bg, wg, do, use_norm, sc, introspection)
            if got is None or "error" in got:
                conv_results.append({"convention": label, "result": got or "fla_not_found"})
                continue

            scale_rels: dict[str, Any] = {}
            for f in GRAD_FIELDS:
                cand_g = got.get(f)
                if cand_g is None:
                    scale_rels[f] = "grad_None"
                else:
                    try:
                        scale_rels[f] = _scale_rel(cand_g.float(), orc_fields[f].float())
                    except Exception as exc:
                        scale_rels[f] = f"error:{exc}"

            numeric_rels = {f: v for f, v in scale_rels.items() if isinstance(v, float)}
            worst = max(numeric_rels.values()) if numeric_rels else float("inf")
            conv_results.append({"convention": label, "scale_rel": scale_rels, "worst": worst})
            print(f"  {label}: worst={worst:.3e}")

            if not best_scale_rel or (
                numeric_rels
                and worst
                < max(
                    (v for v in best_scale_rel.values() if isinstance(v, float)),
                    default=float("inf"),
                )
            ):
                best_scale_rel = scale_rels  # type: ignore[assignment]
                best_kwargs = kw

        # Determinism (2-run bitwise equality using best convention)
        det_result: dict[str, Any] = {"skipped": "no usable convention"}
        if introspection.get("found") is not None:
            try:
                sc_best = best_kwargs.get("scale", dk**-0.5)
                un_best = best_kwargs.get("use_qk_l2norm", True)
                g1 = _try_fla_backward(q, k, v, g, bg, wg, do, un_best, sc_best, introspection)
                g2 = _try_fla_backward(q, k, v, g, bg, wg, do, un_best, sc_best, introspection)
                if g1 and g2 and "error" not in g1 and "error" not in g2:
                    bitwise = all(
                        (g1[f] is None and g2[f] is None)
                        or (g1[f] is not None and g2[f] is not None and torch.equal(g1[f], g2[f]))
                        for f in GRAD_FIELDS
                    )
                    det_result = {"bitwise_equal": bitwise}
                else:
                    det_result = {"error": "one or both runs failed"}
            except Exception as exc:
                det_result = {"error": f"{type(exc).__name__}: {exc}"}

        # Timing with best kwargs
        timing: dict[str, Any] = {"skipped": "no GPU"} if dev.type != "cuda" else {}
        if dev.type == "cuda":
            timing = _time_fla_backward(q, k, v, g, bg, wg, do, introspection, best_kwargs)
            print(f"  timing: {timing.get('median_ms', 'n/a')} ms median")

        per_shape[shape_key] = {
            "shape": [b, ll, h, dk, dv],
            "conventions": conv_results,
            "best_scale_rel": best_scale_rel,
            "determinism": det_result,
            "timing_ms": timing,
        }

    # Overall mapping verdict
    all_numerics = []
    for v_shape in per_shape.values():
        for conv in v_shape.get("conventions", []):
            wr = conv.get("worst")
            if isinstance(wr, float):
                all_numerics.append(wr)
    best_overall = min(all_numerics) if all_numerics else None
    mapping_verdict = (
        "not_established"
        if (best_overall is None or best_overall > ATOL)
        else f"plausible (best_worst={best_overall:.3e})"
    )

    env_block: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "fla_version": fla_version,
    }

    out: dict[str, Any] = {
        "env": env_block,
        "fla_version": fla_version,
        "introspection": introspection,
        "per_shape": per_shape,
        "mapping_notes": {
            "our_params": "q/k/g/b [B,L,H,d_k], v/w [B,L,H,d_v]; b/w already sigmoided; g log-decay key-axis",
            "fla_convention_uncertainty": "fla may apply sigmoid internally to b/w; try as-is",
            "grading_basis": "fp64 reference_gdn2_backward",
            "atol_exploratory": ATOL,
            "mapping_verdict": mapping_verdict,
        },
    }

    if args.out:
        import pathlib

        dest = pathlib.Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=2, default=str))
        print(f"\nwrote {args.out}")
    else:
        print(
            json.dumps({"mapping_verdict": mapping_verdict, "best_overall": best_overall}, indent=2)
        )


if __name__ == "__main__":
    main()
