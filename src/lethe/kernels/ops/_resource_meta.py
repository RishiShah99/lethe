"""Resource-envelope extraction shared by the hand-written Triton kernels.

Walks a ``triton.jit`` function's compilation cache and returns the
*maximum* ``n_regs`` / ``spill_bytes`` / ``shared_bytes`` over every cached
specialisation (dtypes, block sizes, num_warps), in the shape
``gate_res_02_resource_limits`` expects — a conservative envelope, so the
gate checks the worst specialisation rather than whichever cache entry
happens to iterate last. Callers should warm the kernel at the heaviest
shapes they care about before reading. Returns None when nothing has been
compiled yet or the (version-dependent) cache layout has drifted — absence
of evidence must not fabricate evidence.

This module deliberately does not import ``triton``: it only attribute-walks
the jit object handed to it, so it stays importable on CPU-only hosts.
"""

from __future__ import annotations

from typing import Any


def collect_resource_meta(jit_fn: Any) -> dict[str, int] | None:
    """Max-envelope resource metadata across all compiled specialisations."""
    caches = getattr(jit_fn, "device_caches", None)
    compiled: list[Any] = []
    if isinstance(caches, dict):
        for entry in caches.values():
            # 3.x: device_caches[device] is a tuple whose first slot is the
            # signature -> CompiledKernel dict. An empty tuple is cache drift —
            # skip it (indexing would raise, violating the return-None contract).
            cache_dict = entry[0] if isinstance(entry, tuple) and entry else entry
            if isinstance(cache_dict, dict):
                compiled.extend(cache_dict.values())
    legacy = getattr(jit_fn, "cache", None)
    if isinstance(legacy, dict):
        for cache_dict in legacy.values():
            if isinstance(cache_dict, dict):
                compiled.extend(cache_dict.values())

    meta: dict[str, int] | None = None
    for kernel in compiled:
        n_regs = getattr(kernel, "n_regs", None)
        if n_regs is None:
            continue
        if meta is None:
            meta = {"n_regs": 0}
        meta["n_regs"] = max(meta["n_regs"], int(n_regs))
        n_spills = getattr(kernel, "n_spills", None)
        if n_spills is not None:
            # ptxas reports spills in bytes; triton surfaces the raw figure.
            meta["spill_bytes"] = max(meta.get("spill_bytes", 0), int(n_spills))
        shared = getattr(getattr(kernel, "metadata", None), "shared", None)
        if shared is not None:
            meta["shared_bytes"] = max(meta.get("shared_bytes", 0), int(shared))
    return meta


def max_resource_meta(a: dict[str, int] | None, b: dict[str, int] | None) -> dict[str, int] | None:
    """Elementwise-max envelope over two resource metas (None = no evidence).

    Used when dispatch resolves the scan mode by shape (config unset): the
    audited envelope must bound *whichever* kernel could run, so RES-02 sees
    the worst of both instead of the serial one only.
    """
    if a is None:
        return b
    if b is None:
        return a
    return {key: max(a.get(key, 0), b.get(key, 0)) for key in a.keys() | b.keys()}
