"""Staged reward function for RL-discovered Triton kernels."""

from __future__ import annotations

import math

_PARITY_ANCHOR = 1.0
# Floor = _PARITY_ANCHOR + _SLOW_FLOOR_LOG = 0.5; cap = _PARITY_ANCHOR + _FAST_CAP_LOG = 4.0.
_SLOW_FLOOR_LOG = -0.5
_FAST_CAP_LOG = 3.0
_BUG_ROUTING_BONUS = 1.0


def compute_reward(
    *,
    compiled: bool,
    contracts_passed: bool,
    speedup_vs_handwritten: float | None,
    bug_routing: bool = False,
) -> float:
    """Return a scalar reward in [0, 5]."""
    if not compiled:
        return 0.0

    if not contracts_passed:
        return 0.1

    speedup = speedup_vs_handwritten if speedup_vs_handwritten is not None else 0.0
    log_speedup = math.log(speedup) if speedup > 0.0 else _SLOW_FLOOR_LOG
    shaped = min(max(log_speedup, _SLOW_FLOOR_LOG), _FAST_CAP_LOG)
    reward = _PARITY_ANCHOR + shaped

    if bug_routing and speedup > 1.0:
        reward += _BUG_ROUTING_BONUS

    return reward
