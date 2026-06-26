"""Discover the official Mamba-3 SISO training entry point on the box.

The head-to-head bench (claim #3) must time the *crippled* official Mamba-3
backward — the path that launches ``mamba3_siso_bwd_kernel_dqkv`` (named in
state-spaces/mamba#904), which on sm_100 autotunes down to num_warps=2 (TMEM
544 > 512). The existing c2/c6 benches compare against ``selective_scan_fn``
(the Mamba-1 SSD CUDA backward) — the wrong baseline. This script finds the
right one: the autograd.Function / public combined-scan whose backward
dispatches to that kernel, plus its exact call signature, so the bench wires
to a real entry point rather than a guess.

Record-everything mode (mirrors scratch/repro_904.py). Prints + writes JSON.

    CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scratch/mamba3_bwd_discover.py
"""

from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
import re
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any

OUT = Path.home() / "out" / "mamba3_bwd_discover.json"

_NAME_HINTS = re.compile(
    r"siso|chunk_scan|chunk_fwd|combined|selective|mamba3|scan_combined", re.IGNORECASE
)
_KERNEL_NEEDLE = "mamba3_siso_bwd_kernel_dqkv"


def _safe_import(name: str) -> ModuleType | None:
    try:
        return importlib.import_module(name)
    except Exception as exc:
        print(f"  import {name}: FAILED {type(exc).__name__}: {exc}")
        return None


def _walk_packages(root_name: str) -> list[str]:
    """Every importable submodule name under a package root."""
    found: list[str] = []
    root = _safe_import(root_name)
    if root is None or not hasattr(root, "__path__"):
        return found
    found.append(root_name)
    for info in pkgutil.walk_packages(root.__path__, prefix=root_name + "."):
        found.append(info.name)
    return found


def _callables(mod: ModuleType) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for attr in dir(mod):
        if attr.startswith("__"):
            continue
        try:
            obj = getattr(mod, attr)
        except Exception:
            continue
        is_fn = inspect.isfunction(obj) or inspect.isbuiltin(obj)
        is_autograd_fn = inspect.isclass(obj) and any(
            base.__name__ == "Function" for base in inspect.getmro(obj)
        )
        if not (is_fn or is_autograd_fn):
            continue
        if not (_NAME_HINTS.search(attr) or is_autograd_fn):
            continue
        entry: dict[str, Any] = {"kind": "autograd.Function" if is_autograd_fn else "function"}
        try:
            entry["signature"] = str(inspect.signature(obj))
        except (TypeError, ValueError):
            entry["signature"] = None
        if is_autograd_fn:
            for meth in ("forward", "backward"):
                fn = getattr(obj, meth, None)
                if fn is not None:
                    try:
                        entry[f"{meth}_signature"] = str(inspect.signature(fn))
                    except (TypeError, ValueError):
                        entry[f"{meth}_signature"] = None
        out[attr] = entry
    return out


def _grep_source(mod_names: list[str]) -> dict[str, list[str]]:
    """Which installed source files mention the crippled kernel / SISO entry."""
    hits: dict[str, list[str]] = {}
    seen_files: set[str] = set()
    for name in mod_names:
        mod = _safe_import(name)
        src_file = getattr(mod, "__file__", None) if mod else None
        if not src_file or src_file in seen_files:
            continue
        seen_files.add(src_file)
        try:
            text = Path(src_file).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = [
            f"{src_file}:{i}: {ln.strip()}"
            for i, ln in enumerate(text.splitlines(), 1)
            if _KERNEL_NEEDLE in ln
            or re.search(r"def (mamba3_siso|.*chunk_scan_combined|.*siso.*)\(", ln)
            or ("class " in ln and "Fn(" in ln)
        ]
        if lines:
            hits[src_file] = lines[:40]
    return hits


def _inspect_module_forward() -> dict[str, Any]:
    out: dict[str, Any] = {}
    mod = _safe_import("mamba_ssm.modules.mamba3")
    if mod is None:
        out["error"] = "mamba_ssm.modules.mamba3 not importable"
        return out
    cls = getattr(mod, "Mamba3", None)
    if cls is None:
        out["error"] = "Mamba3 class absent"
        return out
    try:
        src = inspect.getsource(cls.forward)
        out["forward_calls"] = sorted(
            {m for m in re.findall(r"\b([a-z_][a-z0-9_]*)\s*\(", src) if _NAME_HINTS.search(m)}
        )
        out["forward_src_head"] = "\n".join(src.splitlines()[:60])
    except (OSError, TypeError) as exc:
        out["error"] = f"getsource failed: {exc}"
    return out


def _try_call_siso_combined(mod_names: list[str]) -> dict[str, Any]:
    """Pure discovery — list which combined entry points exist.

    The caller fills exact args once the signature is known from the dump
    above (the public training fn is ``mamba3_siso_combined``).
    """
    candidates: list[str] = []
    for name in mod_names:
        mod = _safe_import(name)
        if mod is None:
            continue
        for attr in dir(mod):
            if re.search(r"siso.*combined|mamba3.*combined|siso_chunk_scan", attr, re.IGNORECASE):
                candidates.append(f"{name}.{attr}")
    return {"candidate_names": candidates}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {}

    mamba = _safe_import("mamba_ssm")
    report["mamba_ssm_version"] = getattr(mamba, "__version__", "unknown") if mamba else None
    report["mamba_ssm_file"] = getattr(mamba, "__file__", None) if mamba else None

    triton_mods = _walk_packages("mamba_ssm.ops.triton")
    tilelang_mods = _walk_packages("mamba_ssm.ops.tilelang")
    module_mods = _walk_packages("mamba_ssm.modules")
    report["submodules"] = {
        "ops.triton": triton_mods,
        "ops.tilelang": tilelang_mods,
        "modules": module_mods,
    }

    entry_points: dict[str, dict[str, Any]] = {}
    for name in triton_mods + tilelang_mods + module_mods:
        mod = _safe_import(name)
        if mod is None:
            continue
        found = _callables(mod)
        if found:
            entry_points[name] = found
    report["entry_points"] = entry_points

    report["kernel_grep"] = _grep_source(triton_mods + tilelang_mods)
    report["mamba3_module_forward"] = _inspect_module_forward()

    try:
        report["call_probe"] = _try_call_siso_combined(triton_mods)
    except Exception:
        report["call_probe"] = {"error": traceback.format_exc()}

    OUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("\n=== SUMMARY ===")
    print("mamba_ssm:", report["mamba_ssm_version"])
    print("triton submodules:", len(triton_mods))
    print("entry-point modules with hits:", list(entry_points.keys()))
    print("kernel-grep files:", list(report["kernel_grep"].keys()))
    print("Mamba3.forward SSM-core calls:", report["mamba3_module_forward"].get("forward_calls"))
    print(f"\nfull dump -> {OUT}")


if __name__ == "__main__":
    main()
