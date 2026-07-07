"""Introspect cutlass-dsl 4.5.2 for the primitives inc-B2 needs (no GPU compute).

The keystone B2 mechanic is filling an MMA SMEM operand from a non-TMA source
(register/compute value) into the make_smem_layout_a swizzle, then having the MMA
read it. This probes the available copy/partition helpers + the tcgen05 operand
sources so the smoke is written against real APIs, not guesses.
"""

from __future__ import annotations

import inspect


def names(mod: object, needle: str = "") -> list[str]:
    out = []
    for n in dir(mod):
        if n.startswith("_"):
            continue
        if needle and needle.lower() not in n.lower():
            continue
        out.append(n)
    return out


def sig(fn: object) -> str:
    try:
        return str(inspect.signature(fn))  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return "<no sig>"


def main() -> None:
    import cutlass.cute as cute
    import cutlass.utils as utils
    import cutlass.utils.blackwell_helpers as bh
    from cutlass.cute.nvgpu import cpasync, tcgen05

    print("== cute: copy/tiled/partition ==")
    for n in names(cute):
        if any(k in n.lower() for k in ("copy", "tiled", "partition", "autovec", "fragment", "rmem")):
            obj = getattr(cute, n)
            print(f"  cute.{n} {sig(obj) if callable(obj) else ''}")

    print("\n== blackwell_helpers ==")
    for n in names(bh):
        obj = getattr(bh, n)
        print(f"  bh.{n} {sig(obj) if callable(obj) else ''}")

    print("\n== utils (copy/tiled/tmem) ==")
    for n in names(utils):
        if any(k in n.lower() for k in ("copy", "tiled", "tmem", "smem", "layout")):
            obj = getattr(utils, n)
            print(f"  utils.{n} {sig(obj) if callable(obj) else ''}")

    print("\n== tcgen05 (operand source / mma / ts) ==")
    for n in names(tcgen05):
        if any(k in n.lower() for k in ("operand", "source", "mma", "ts", "copy", "tmem", "ld", "st", "field")):
            print(f"  tcgen05.{n}")
    print("  OperandSource members:", names(tcgen05.OperandSource) if hasattr(tcgen05, "OperandSource") else "n/a")

    print("\n== cpasync ==")
    for n in names(cpasync):
        print(f"  cpasync.{n}")

    print("\n== make_tiled_copy* signatures ==")
    for cand in ("make_tiled_copy_A", "make_tiled_copy_B", "make_tiled_copy_C",
                 "make_tiled_copy_tv", "make_tiled_copy", "make_copy_atom", "autovec_copy"):
        fn = getattr(cute, cand, None)
        print(f"  cute.{cand}: {sig(fn) if fn else 'MISSING'}")
    for cand in ("make_tiled_copy_A", "make_tiled_copy_B"):
        fn = getattr(bh, cand, None) or getattr(utils, cand, None)
        print(f"  helper.{cand}: {sig(fn) if fn else 'MISSING'}")


if __name__ == "__main__":
    main()
