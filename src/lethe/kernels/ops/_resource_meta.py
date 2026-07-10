"""Resource-envelope extraction shared by the hand-written Triton kernels."""

from __future__ import annotations

from typing import Any


def collect_resource_meta(jit_fn: Any) -> dict[str, int] | None:
    """Max-envelope resource metadata across all compiled specialisations."""
    caches = getattr(jit_fn, "device_caches", None)
    compiled: list[Any] = []
    if isinstance(caches, dict):
        for entry in caches.values():
            # 3.x: device_caches[device] is a tuple whose first slot is the signature -> CompiledKernel dict.
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
    """Elementwise-max envelope over two resource metas (None = no evidence)."""
    if a is None:
        return b
    if b is None:
        return a
    return {key: max(a.get(key, 0), b.get(key, 0)) for key in a.keys() | b.keys()}
