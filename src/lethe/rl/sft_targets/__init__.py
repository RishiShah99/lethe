"""Verified warm-start targets: one self-contained eager solution per curriculum op."""

from pathlib import Path

# Two variants per op: eager lands the 0.5 contract floor, triton seeds speedup exploration.
_OPS: tuple[str, ...] = (
    "forward_chunked_scan",
    "backward_selective_scan",
    "mimo_backward",
    "complex_scan_rope",
    "fused_block_forward",
    "fused_block_backward",
)

_VARIANT_SUFFIX: dict[str, str] = {"eager": "", "triton": "_triton"}


def available_targets() -> tuple[str, ...]:
    return _OPS


def target_variants(op_name: str) -> tuple[str, ...]:
    if op_name not in _OPS:
        raise KeyError(op_name)
    return tuple(
        v
        for v, suffix in _VARIANT_SUFFIX.items()
        if (Path(__file__).parent / f"{op_name}{suffix}.py").exists()
    )


def target_source(op_name: str, variant: str = "eager") -> str:
    """Return the verified candidate source for *(op_name, variant)*."""
    if op_name not in _OPS:
        raise KeyError(op_name)
    path = Path(__file__).parent / f"{op_name}{_VARIANT_SUFFIX[variant]}.py"
    return path.read_text(encoding="utf-8")
