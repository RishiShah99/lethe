"""Bridge between the verifier outputs and the GRPO reward signal."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from lethe.rl.rollout import Candidate, ScoredCandidate
from lethe.verifier.contracts import GateResult, run_all_gates
from lethe.verifier.reward import compute_reward

# Gates that are implemented today.
_IMPLEMENTED_GATES: tuple[str, ...] = (
    "gate_cmp_01_input_variation",
    "gate_cmp_02_gradient_correctness",
    "gate_cmp_03_shape_polymorphism",
    "gate_ord_01_reduction_order_tolerance",
    "gate_ord_02_atomic_determinism",
    "gate_ord_03_noncommutative_reduction",
    "gate_prc_01_precision_regime",
    "gate_prc_02_mixed_precision_accumulation",
    "gate_exc_01_exceptional_values",
    "gate_exc_02_subnormal_handling",
    "gate_res_01_memory_residency",
)


def _contracts_passed(results: dict[str, GateResult]) -> bool:
    """All implemented gates must pass (stubbed gates don't count)."""
    return all(results[name].passed for name in _IMPLEMENTED_GATES)


def score_callable(
    candidate: Candidate,
    candidate_fn: Callable[..., torch.Tensor],
    reference_fn: Callable[..., torch.Tensor],
    *,
    compiled: bool = True,
    speedup_vs_handwritten: float | None = None,
    bug_routing: bool = False,
    gate_kwargs: dict[str, Any] | None = None,
) -> ScoredCandidate:
    """Score a candidate against the reference and return a ``ScoredCandidate``."""
    if not compiled:
        return ScoredCandidate(
            candidate=candidate,
            reward=compute_reward(
                compiled=False,
                contracts_passed=False,
                speedup_vs_handwritten=None,
            ),
            compiled=False,
            contracts_passed=False,
            speedup_vs_handwritten=None,
            bug_routing=False,
            gate_results={},
        )

    results = run_all_gates(candidate_fn, reference_fn, **(gate_kwargs or {}))
    passed = _contracts_passed(results)
    reward = compute_reward(
        compiled=True,
        contracts_passed=passed,
        speedup_vs_handwritten=speedup_vs_handwritten,
        bug_routing=bug_routing,
    )
    return ScoredCandidate(
        candidate=candidate,
        reward=reward,
        compiled=True,
        contracts_passed=passed,
        speedup_vs_handwritten=speedup_vs_handwritten,
        bug_routing=bug_routing,
        gate_results={name: results[name].passed for name in results},
    )
