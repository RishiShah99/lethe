"""Tunable kernel configurations — the RL-as-autotuner action space.

The from-scratch source-generation policy is capped at imitating its SFT
target (the hand-written kernel), so its measured speedup is ~1.0 by
construction. The action space here is instead a *configuration* applied to
the trusted, already-correct in-repo kernel: only knobs that are
correctness-invariant (num_warps, num_stages, the D/P tiling, the checkpoint
granularity chunk_k) vary; the knobs a kernel needs for *correctness*
(block_n holding the full state dim, block_k holding the conv window) are
pinned to the shape and never appear here. A config can therefore only ever
change *performance* — or fail to compile / spill past budget / OOM, which
the verifier scores honestly. Correctness is still re-gated per config:
num_warps and the tiling shift reduction/FMA order within the contract
tolerances, and the gate battery confirms each config stays inside them.

``SEARCH_GRID`` enumerates the searched knobs per op — both the
exhaustive-autotuning ceiling and the legal action set the policy samples
from. A ``None`` field means "use the kernel's own default heuristic for
this knob"; the all-``None`` config is the shipped default and is always
valid.
"""

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
    """A point in a kernel's launch-knob search space (None = shipped default).

    All knobs are correctness-invariant. ``scan_mode`` selects the SISO scan
    algorithm (forward and backward) — ``"serial"`` (the O(L) walk, the
    byte-identical default) or ``"chunk_parallel"`` (the SSD chunked-carry
    reassociation, the long-L lever); ``chunk_len`` is its chunk-parallel
    granularity (must divide L), a no-op in serial mode. Both reassociate the
    *same* recurrence, so the gate battery re-confirms correctness within the
    eps*sqrt(chain)*scale band per config.
    """

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


# Per-op searched knobs. block_d tiles D (forward/conv ops); block_p tiles
# headdim (the head-structured ops); chunk_k is the in-chunk recompute window
# of the checkpointed backward ops. block_n / block_k are absent by design —
# they are correctness constraints, not knobs.
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
        "block_p": _BLOCK,
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
    """Return the config's grid/shape violations (empty list = legal).

    A knob set on the wrong op, or to a value outside its grid, is a
    violation. With ``shape`` given, ``chunk_k`` and ``chunk_len`` must each
    divide ``seq_len`` (the checkpoint / chunk-parallel window tiles the
    sequence).
    """
    if op not in SEARCH_GRID:
        return [f"no search grid for op {op!r}"]
    grid = SEARCH_GRID[op]
    violations: list[str] = []
    for name, value in config.searched().items():
        if name not in grid:
            violations.append(f"{name} is not tunable for {op}")
        elif value not in grid[name]:
            violations.append(f"{name}={value} not in {grid[name]}")
    if shape is not None and config.chunk_k is not None and shape.seq_len % config.chunk_k != 0:
        violations.append(f"chunk_k={config.chunk_k} does not divide seq_len={shape.seq_len}")
    if shape is not None and config.chunk_len is not None and shape.seq_len % config.chunk_len != 0:
        violations.append(f"chunk_len={config.chunk_len} does not divide seq_len={shape.seq_len}")
    return sorted(violations)


def iter_configs(op: str) -> Iterator[KernelConfig]:
    """Yield every config in *op*'s exhaustive grid (the autotuning ceiling)."""
    grid = SEARCH_GRID[op]
    keys = list(grid)
    for combo in itertools.product(*(grid[k] for k in keys)):
        kwargs: dict[str, Any] = dict(zip(keys, combo, strict=True))
        yield KernelConfig(**kwargs)


def make_configured_op(op: str, config: KernelConfig) -> Callable[..., Any]:
    """A public-signature callable for *op* with *config* bound into the launcher.

    The returned callable IS the trusted in-repo public op with the kernel
    config bound in — the gate battery and the speedup bench invoke it exactly
    as they invoke the shipped op, the only difference being the launch knobs.
    No candidate source is involved, so the source-gaming screens the
    generation path needs do not apply. The ops are imported lazily so the pure
    config types stay importable without torch/triton.
    """
    if op not in SEARCH_GRID:
        raise KeyError(f"{op!r} is not a tunable op")
    from flash_mamba_rl.kernels import ops as hand_ops

    return functools.partial(getattr(hand_ops, op), config=config)
