"""Staged reward function for RL-discovered Triton kernels.

Reward schedule (per project DESIGN spec):
  - not compiled          → 0.0
  - compiled, failed contracts → 0.1
  - compiled, passed, speedup ≤ 1.0 → 0.5
  - compiled, passed, speedup > 1.0 → 1.0 + clip(log(speedup), 0, 3)
  - compiled, passed, speedup > 1.0, bug_routing=True
      (kernel compiles on Blackwell where hand-written fails)
                          → 2.0 + clip(log(speedup), 0, 3)
"""

from __future__ import annotations

import math


def compute_reward(
    *,
    compiled: bool,
    contracts_passed: bool,
    speedup_vs_handwritten: float | None,
    bug_routing: bool = False,
) -> float:
    """Return a scalar reward in [0, 5].

    Parameters
    ----------
    compiled:
        Whether the kernel compiled without errors.
    contracts_passed:
        Whether all implemented Kernel Contract gates passed.
    speedup_vs_handwritten:
        ``t_reference / t_candidate``; >1 means candidate is faster.
        Pass ``None`` if timing was not measured.
    bug_routing:
        True when the candidate compiles successfully on a Blackwell GPU where
        the hand-written reference triggers the ptxas C7907 ICE.  Signals the
        RL agent discovered a bug-routing strategy.

    Returns
    -------
    float
        Scalar reward.
    """
    if not compiled:
        return 0.0

    if not contracts_passed:
        return 0.1

    speedup = speedup_vs_handwritten if speedup_vs_handwritten is not None else 0.0

    if speedup <= 1.0:
        return 0.5

    # speedup > 1.0 — kernel is strictly faster
    log_speedup_clipped = min(max(math.log(speedup), 0.0), 3.0)

    if bug_routing:
        return 2.0 + log_speedup_clipped

    return 1.0 + log_speedup_clipped
