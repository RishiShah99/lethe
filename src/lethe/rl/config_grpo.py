"""Config-emitting GRPO: the action is a KernelConfig JSON, not source."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import fields
from typing import Any

from lethe.kernels.autotune import SEARCH_GRID, KernelConfig, ShapeSpec
from lethe.verifier.candidate_scoring import (
    DEFAULT_EXCLUDE_GATES,
    score_candidate_config,
)

_JSON_BLOCK = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)
_CONFIG_FIELDS = frozenset(f.name for f in fields(KernelConfig))


def extract_config(completion: str) -> str | None:
    """The final fenced block in *completion*; falls back to the bare body."""
    blocks = _JSON_BLOCK.findall(completion)
    if blocks:
        return str(blocks[-1]).strip()
    stripped = completion.strip()
    return stripped or None


def parse_config(text: str) -> KernelConfig | None:
    """Parse a config JSON string into a ``KernelConfig``, or None if illegal."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict) or not (obj.keys() <= _CONFIG_FIELDS):
        return None
    for key, value in obj.items():
        if value is None:
            continue
        if key == "scan_mode":
            if not isinstance(value, str):
                return None
        # bool is an int subclass; a JSON true/false is not a launch knob.
        elif isinstance(value, bool) or not isinstance(value, int):
            return None
    return KernelConfig(**obj)


def _illegal_action_result(error: str) -> dict[str, Any]:
    """A 0.0-reward score dict for an unparseable / illegal config emission."""
    return {
        "status": "illegal_config",
        "error": error,
        "compiled": False,
        "contracts_passed": False,
        "reward": 0.0,
        "gates": {},
        "views_passed": 0,
        "views_total": 0,
        "first_failed_view": None,
    }


def score_config_candidate(
    text: str,
    *,
    op: str,
    device: str = "cuda",
    shape: ShapeSpec | None = None,
    timeout_s: float = 300.0,
    exclude_gates: tuple[str, ...] = DEFAULT_EXCLUDE_GATES,
    reward_shaping: str = "none",
    measure_speedup: bool = True,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Parse and score one config emission against ``shape``; illegal emissions score 0.0."""
    cfg = parse_config(text)
    if cfg is None:
        return _illegal_action_result(f"unparseable or illegal config: {text[:120]!r}")
    return score_candidate_config(
        cfg,
        op=op,
        device=device,
        shape=shape,
        timeout_s=timeout_s,
        exclude_gates=exclude_gates,
        reward_shaping=reward_shaping,
        measure_speedup=measure_speedup,
        extra_env=extra_env,
    )


def build_config_scorer(
    op: str,
    shape: ShapeSpec,
    *,
    device: str = "cuda",
    timeout_s: float = 300.0,
    measure_speedup: bool = True,
    extra_env: dict[str, str] | None = None,
) -> Callable[[str], dict[str, Any]]:
    """A :class:`GRPOTrainingLoop` ``scorer`` closure for (op, shape) configs."""

    def scorer(text: str) -> dict[str, Any]:
        return score_config_candidate(
            text,
            op=op,
            shape=shape,
            device=device,
            timeout_s=timeout_s,
            measure_speedup=measure_speedup,
            extra_env=extra_env,
        )

    return scorer


def serial_seed_completions(op: str, n: int) -> list[str]:
    """``n`` well-formed ``scan_mode="serial"`` emissions for *op*; ``[]`` if no such knob."""
    if n <= 0:
        return []
    grid = SEARCH_GRID.get(op, {})
    if "scan_mode" not in grid:
        return []
    warps: list[int] = [int(w) for w in grid.get("num_warps", ())]
    out: list[str] = []
    for i in range(n):
        cfg: dict[str, int | str] = {"scan_mode": "serial"}
        if warps:
            cfg["num_warps"] = warps[i % len(warps)]
        out.append(f"```json\n{json.dumps(cfg)}\n```")
    return out
