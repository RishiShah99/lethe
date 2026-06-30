"""Introspect the CuTe DSL launch/stream API — how to issue a compiled launch on a
specific CUDA stream (the prerequisite for CUDA-graph capture; #47 found the launch
lands on a non-capture stream → empty graph).

  PYTHONPATH=src:. ~/cuteenv/bin/python scratch/stream_api_probe.py
"""

import inspect

import cutlass
import cutlass.cute as cute

print("=== cute.compile ===")
print("sig:", inspect.signature(cute.compile))

# The compiled executable type
try:
    from cutlass.base_dsl import jit_executor as je

    print("\n=== jit_executor members ===")
    print([m for m in dir(je) if not m.startswith("__")])
    for name in dir(je):
        obj = getattr(je, name)
        if inspect.isclass(obj) and "Exec" in name:
            print(f"\n{name}:")
            for meth in ("__call__", "launch", "run"):
                if hasattr(obj, meth):
                    try:
                        print(f"  {meth}{inspect.signature(getattr(obj, meth))}")
                    except (ValueError, TypeError):
                        print(f"  {meth}: <no sig>")
except Exception as e:
    print("jit_executor introspection failed:", repr(e))

print("\n=== kernel .launch signature ===")
# A LaunchConfig / launch op carries grid/block/stream. Find where stream lives.
for modname in ("cutlass.cute", "cutlass.cute.nvgpu", "cutlass._mlir"):
    try:
        mod = __import__(modname, fromlist=["x"])
        cands = [m for m in dir(mod) if "launch" in m.lower() or "Launch" in m]
        if cands:
            print(f"{modname}: {cands}")
    except Exception as e:
        print(f"{modname}: {e!r}")

print("\n=== cutlass.cuda / stream helpers ===")
for modname in ("cutlass.cuda", "cutlass.utils"):
    try:
        mod = __import__(modname, fromlist=["x"])
        print(f"{modname}:", [m for m in dir(mod) if "tream" in m or "current" in m.lower()])
    except Exception as e:
        print(f"{modname}: {e!r}")

# Does the runtime read a current stream we can set? cuda.bindings driver current stream.
print("\n=== cuda bindings ===")
try:
    from cuda import bindings  # type: ignore

    print("cuda.bindings present:", bindings.__name__)
except Exception as e:
    print("cuda.bindings:", repr(e))
try:
    import cuda.cuda as ccuda  # type: ignore

    print("cuda.cuda present")
    _ = ccuda
except Exception as e:
    print("cuda.cuda:", repr(e))

print("\nver cutlass:", getattr(cutlass, "__version__", "?"))
