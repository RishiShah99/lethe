"""Tunable kernel configurations, the RL-as-autotuner action space."""

from __future__ import annotations

import functools
import itertools
from collections.abc import Callable, Iterator
from dataclasses import dataclass, fields
from typing import Any

_WARPS = (2, 4, 8)
_STAGES = (1, 2, 3)
_BLOCK = (16, 32, 64, 128)
_CHUNK_K = (1, 2, 4, 8, 16)
_SCAN_MODE = ("serial", "chunk_parallel")
_CHUNK_LEN = (64, 128, 256, 512)


@dataclass(frozen=True)
class KernelConfig:
    """A point in a kernel's launch-knob search space (None = shipped default)."""

    block_d: int | None = None
    block_p: int | None = None
    chunk_k: int | None = None
    num_warps: int | None = None
    num_stages: int | None = None
    scan_mode: str | None = None
    chunk_len: int | None = None

    def searched(self) -> dict[str, int | str]:
        """The explicitly-set knobs, dropping the None (default) fields."""
        return {f.name: v for f in fields(self) if (v := getattr(self, f.name)) is not None}


@dataclass(frozen=True)
class ShapeSpec:
    """A bench/target shape, for shape-dependent config validation and scoring."""

    batch: int
    seq_len: int
    width: int
    n_state: int | None = None


# Per-op searched knobs.
SEARCH_GRID: dict[str, dict[str, tuple[int, ...] | tuple[str, ...]]] = {
    "forward_chunked_scan": {
        "block_d": _BLOCK,
        "num_warps": _WARPS,
        "num_stages": _STAGES,
        "scan_mode": _SCAN_MODE,
        "chunk_len": _CHUNK_LEN,
    },
    "backward_selective_scan": {
        "block_d": _BLOCK,
        "chunk_k": _CHUNK_K,
        "num_warps": _WARPS,
        "num_stages": _STAGES,
        "scan_mode": _SCAN_MODE,
        "chunk_len": _CHUNK_LEN,
    },
    "mimo_backward": {
        "chunk_k": _CHUNK_K,
        "num_warps": _WARPS,
        "num_stages": _STAGES,
    },
    "complex_scan_rope": {"block_p": _BLOCK, "num_warps": _WARPS, "num_stages": _STAGES},
    "fused_block_forward": {"block_d": _BLOCK, "num_warps": _WARPS, "num_stages": _STAGES},
    "fused_block_backward": {
        "block_d": _BLOCK,
        "chunk_k": _CHUNK_K,
        "num_warps": _WARPS,
        "num_stages": _STAGES,
        "scan_mode": _SCAN_MODE,
        "chunk_len": _CHUNK_LEN,
    },
}


def tunable_ops() -> tuple[str, ...]:
    return tuple(SEARCH_GRID)


def grid_size(op: str) -> int:
    """Number of configs in the exhaustive grid for *op*."""
    n = 1
    for values in SEARCH_GRID[op].values():
        n *= len(values)
    return n


def validate(op: str, config: KernelConfig, *, shape: ShapeSpec | None = None) -> list[str]:
    """Return the config's grid/shape violations (empty list = legal)."""
    if op not in SEARCH_GRID:
        return [f"no search grid for op {op!r}"]
    grid = SEARCH_GRID[op]
    violations: list[str] = []
    for name, value in config.searched().items():
        if name not in grid:
            violations.append(f"{name} is not tunable for {op}")
        elif value not in grid[name]:
            violations.append(f"{name}={value} not in {grid[name]}")
    if shape is not None and config.chunk_k is not None:
        if config.chunk_k <= 0:
            violations.append(f"chunk_k={config.chunk_k} must be positive")
        elif shape.seq_len % config.chunk_k != 0:
            violations.append(f"chunk_k={config.chunk_k} does not divide seq_len={shape.seq_len}")
    if shape is not None and config.chunk_len is not None:
        if config.chunk_len <= 0:
            violations.append(f"chunk_len={config.chunk_len} must be positive")
        elif shape.seq_len % config.chunk_len != 0:
            violations.append(
                f"chunk_len={config.chunk_len} does not divide seq_len={shape.seq_len}"
            )
    return sorted(violations)


def iter_configs(op: str) -> Iterator[KernelConfig]:
    """Yield every config in *op*'s exhaustive grid (the autotuning ceiling)."""
    grid = SEARCH_GRID[op]
    keys = list(grid)
    for combo in itertools.product(*(grid[k] for k in keys)):
        kwargs: dict[str, Any] = dict(zip(keys, combo, strict=True))
        yield KernelConfig(**kwargs)


def make_configured_op(op: str, config: KernelConfig) -> Callable[..., Any]:
    """A public-signature callable for *op* with *config* bound into the launcher."""
    if op not in SEARCH_GRID:
        raise KeyError(f"{op!r} is not a tunable op")
    from lethe.kernels import ops as hand_ops

    return functools.partial(getattr(hand_ops, op), config=config)
