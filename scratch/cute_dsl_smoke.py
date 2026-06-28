"""CuTe DSL toolchain de-risk + API probe (Phase-2, before investing in K#1/K#2).

Proves the box can JIT-compile and run a Python CuTe DSL kernel for sm_100 (B200),
and discovers the exact API surface of the installed cutlass version (paths for the
sm100 MMA helpers, the cute.Tensor element I/O) so the real kernels use correct syntax.

Imports are module-level: the DSL resolves @cute.jit/@cute.kernel annotations with
``inspect.signature(..., eval_str=True)`` against the function's module globals, so
``cute`` must live there (a nested import raises NameError at compile).

Writes results/cute_dsl_smoke.json. GO iff imports + sm_100 device + the hello kernel
compiles and runs. Run on the box: uv run --no-sync python scratch/cute_dsl_smoke.py
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import traceback
from pathlib import Path
from typing import Any

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05  # noqa: F401
from cutlass.cute.runtime import from_dlpack


def _imports() -> dict[str, Any]:
    return {"ok": True, "cutlass_version": getattr(cutlass, "__version__", "unknown")}


def _device() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda": False}
    major, minor = torch.cuda.get_device_capability(0)
    return {
        "cuda": True,
        "name": torch.cuda.get_device_name(0),
        "cc": f"sm_{major}{minor}",
        "is_sm100": major == 10,
    }


@cute.kernel
def _hello_kernel() -> None:
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        cute.printf("cute-dsl hello from sm_100\n")


@cute.jit
def _hello_host() -> None:
    _hello_kernel().launch(grid=(1, 1, 1), block=(32, 1, 1))


def _hello() -> dict[str, Any]:
    cute.compile(_hello_host)()
    torch.cuda.synchronize()
    return {"compiled_and_ran": True}


@cute.kernel
def _vadd_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    i = bidx * bdim + tidx
    if i < cute.size(gA):
        gC[i] = gA[i] + gB[i]


@cute.jit
def _vadd_host(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor) -> None:
    threads = 256
    blocks = (cute.size(gA) + threads - 1) // threads
    _vadd_kernel(gA, gB, gC).launch(grid=(blocks, 1, 1), block=(threads, 1, 1))


def _vadd() -> dict[str, Any]:
    n = 1 << 16
    a = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)
    c = torch.empty_like(a)
    cute.compile(_vadd_host, from_dlpack(a), from_dlpack(b), from_dlpack(c))(
        from_dlpack(a), from_dlpack(b), from_dlpack(c)
    )
    torch.cuda.synchronize()
    err = (c - (a + b)).abs().max().item()
    return {"ran": True, "max_err": err, "passed": err == 0.0}


def _probe_api() -> dict[str, Any]:
    """Locate the sm100 MMA helpers + record the cute.Tensor / arch surface."""
    out: dict[str, Any] = {}
    candidates = [
        "cutlass.utils.sm100_utils",
        "cutlass.cute.nvgpu.sm100_utils",
        "cutlass.utils.blackwell_helpers",
        "cutlass.cute.nvgpu.warpgroup",
        "cutlass.cute.nvgpu.cpasync",
        "cutlass.pipeline",
        "cutlass.cute.nvgpu.tcgen05",
    ]
    found: dict[str, Any] = {}
    for mod in candidates:
        try:
            spec = importlib.util.find_spec(mod)
        except (ImportError, ModuleNotFoundError, ValueError):
            spec = None
        if spec is None:
            found[mod] = None
            continue
        try:
            m = importlib.import_module(mod)
            names = [n for n in dir(m) if not n.startswith("_")]
            hit = [n for n in names if "mma" in n.lower() or "tiled" in n.lower()]
            found[mod] = {"present": True, "mma_like": hit[:20], "n_public": len(names)}
        except Exception as exc:  # noqa: BLE001
            found[mod] = {"present": True, "import_error": f"{type(exc).__name__}: {exc}"}
    out["modules"] = found

    # scan the cutlass package tree for any module mentioning sm100 / blackwell.
    try:
        import pkgutil

        root = importlib.import_module("cutlass")
        hits = []
        for info in pkgutil.walk_packages(root.__path__, prefix="cutlass."):
            low = info.name.lower()
            if "sm100" in low or "blackwell" in low:
                hits.append(info.name)
        out["sm100_modules"] = hits[:40]
    except Exception as exc:  # noqa: BLE001
        out["sm100_modules"] = f"{type(exc).__name__}: {exc}"

    out["cute_public"] = [n for n in dir(cute) if not n.startswith("_")][:60]
    out["cute_arch_public"] = [n for n in dir(cute.arch) if not n.startswith("_")][:60]
    return out


def main() -> None:
    out: dict[str, Any] = {}
    steps = (
        ("imports", _imports),
        ("device", _device),
        ("hello", _hello),
        ("vadd", _vadd),
        ("api", _probe_api),
    )
    for name, fn in steps:
        try:
            out[name] = fn()
        except Exception as exc:  # noqa: BLE001
            out[name] = {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()}

    go = (
        isinstance(out.get("imports"), dict)
        and out["imports"].get("ok")
        and isinstance(out.get("device"), dict)
        and out["device"].get("is_sm100")
        and isinstance(out.get("hello"), dict)
        and out["hello"].get("compiled_and_ran")
    )
    out["GO"] = bool(go)

    dest = Path("results/cute_dsl_smoke.json")
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))
    print(f"\nGO={out['GO']}  ->  {dest}")


if __name__ == "__main__":
    main()
