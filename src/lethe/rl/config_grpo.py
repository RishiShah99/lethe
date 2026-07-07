"""E2.c — config-emitting GRPO: the action is a KernelConfig JSON, not source.

The from-scratch source policy is capped at parity by construction — it can
only imitate its SFT target, the hand-written kernel, so its measured speedup
is ~1.0. Here the policy instead emits a JSON ``KernelConfig`` of
correctness-invariant launch knobs (num_warps, num_stages, the D/P tiling, the
checkpoint chunk_k) applied to the trusted in-repo kernel, scored at a target
shape by :func:`score_candidate_config`. The shipped default is tuned only near
the training shape, so an off-default / long-L target leaves real headroom (the
E2.d grid sweep measures 1.5-1.7x on fused_block_forward).

This reuses :class:`GRPOTrainingLoop` verbatim through its ``extractor`` and
``scorer`` hooks — the same advantages / clipped-surrogate+KL update /
degenerate-skip / checkpoint machinery. Only the action representation changes:
a few-token JSON config instead of a multi-thousand-token kernel, which removes
the source track's generation-throughput and backward-truncation failure modes
at once.

A candidate that does not parse as a legal config (bad JSON, a hallucinated
knob, a non-int value) is an illegal action scored 0.0 — the grading of
untrusted policy output, the analog of the source track's no-code /
forbidden-import zero. Grid membership and the ``chunk_k | seq_len``
divisibility are NOT checked here: :func:`lethe.kernels.autotune.validate`
(inside :func:`score_candidate_config`) is the single legality oracle, demoting
an out-of-grid config to the ``invalid_config`` 0.0.
"""

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
    """The final fenced block in *completion*; falls back to the bare body.

    The prompt asks for a fenced ```json block, but a config is tiny enough
    that a policy may emit a bare object with no fence — so when there is no
    fenced block the stripped completion is returned for the parser to try.
    None only when there is nothing non-empty to parse. Consequence for the
    loop: a non-empty-but-unparseable completion is NOT a ``no_code_block`` (it
    reaches the scorer and lands at the ``illegal_config`` 0.0), so on this
    track ``n_no_code`` counts only genuinely empty completions.
    """
    blocks = _JSON_BLOCK.findall(completion)
    if blocks:
        return str(blocks[-1]).strip()
    stripped = completion.strip()
    return stripped or None


def parse_config(text: str) -> KernelConfig | None:
    """Parse a config JSON string into a ``KernelConfig``, or None if illegal.

    Enforces only the JSON *shape* — a dict over ``KernelConfig``'s fields with
    int (not bool) or null values, except ``scan_mode`` which is a string; the
    empty object is the shipped default. Grid membership and shape divisibility
    are left to ``autotune.validate`` at scoring time (one legality oracle, no
    duplicated grid here).
    """
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
    """A 0.0-reward score dict for an unparseable / illegal config emission.

    Mirrors ``candidate_scoring._failure``'s shape so rollout rows stay uniform
    across the source and config tracks.
    """
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
    """Parse one config emission and score it; illegal emissions score 0.0.

    ``shape`` is the speedup bench shape; only its (batch, seq_len, width) reach
    the bench — ``n_state`` is fixed per op by the harness aux and ignored here.
    """
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
    """``n`` well-formed ``scan_mode="serial"`` emissions for *op* (``[]`` if it
    has no ``scan_mode`` knob).

    Forced-exploration seeds for the config-RL group (#14). A fresh 32B prior
    commits to ``chunk_parallel`` at every shape, so a sampled group never
    contains a serial config and GRPO has no gradient toward serial — it cannot
    learn the saturated-shape crossover the boundary sweep proved exists
    (``results/scan_mode_boundary.json``). Replacing the first ``n`` of the K
    sampled completions with these makes the reward see serial: at a saturated
    shape serial out-scores chunk_parallel so its group-relative advantage is
    positive (policy pushed toward serial); at long-L it loses so the advantage
    is negative (pushed away). That contrast is the learned boundary. The
    injection is GRPO-sound because the loop sets ``old_lp = new_lp.detach()``,
    so the step-0 importance ratio is 1 even for these off-policy samples.

    Each seed pins a distinct ``num_warps`` from the op's grid (serial's main
    perf knob) so the scorer can surface the best serial config rather than
    betting on one; all are grid-legal, so ``autotune.validate`` passes them.
    """
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
