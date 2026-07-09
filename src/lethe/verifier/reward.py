"""Staged reward function for RL-discovered Triton kernels.

Reward schedule:
  - not compiled                          → 0.0
  - compiled, contracts failed            → 0.1
  - compiled, contracts passed            → 1.0 + clip(log(speedup), -0.5, 3.0)
      speedup None / 0 / ≤ e**-0.5≈0.61   → 0.5   (floor: correct but slow)
      speedup = 1.0 (parity)              → 1.0
      speedup ≥ e**3 ≈ 20.1               → 4.0   (cap)
  - + 1.0 bug-routing bonus when contracts passed AND speedup > 1.0 on a
    Blackwell GPU where the hand-written kernel triggers the TMEM overflow.

The contracts-passed region is continuous and monotone in speedup: the old
table paid a flat 0.5 for any speedup ≤ 1 and jumped to 1.0+ above it, so
0.9x and 0.99x scored identically and a candidate hovering at parity landed
on either side of a 0.5-wide cliff by timing noise; no gradient pulled the
policy *toward* faster from below. Anchoring on log(speedup) through the
contract floor removes the cliff (parity scores 1.0, approached smoothly)
and gives a gradient across the whole sub-parity band down to the 0.5 floor.

The floor stays at 0.5, so contracts-passed always beats contracts-failed
(0.1) by a clear margin and the speedup term still only ever *adds* above a
correct kernel; speedup is paid strictly after contracts pass, never before
(a project invariant; the s>1 branch and the bug-routing bonus are unchanged
from the prior table, so recorded faster-than-parity rewards are unaffected).
"""

from __future__ import annotations

import math

_PARITY_ANCHOR = 1.0
# Floor = _PARITY_ANCHOR + _SLOW_FLOOR_LOG = 0.5 (the prior "correct but slow"
# reward); _FAST_CAP_LOG caps the fastest credit at 1.0 + 3.0 = 4.0.
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
    """Return a scalar reward in [0, 5].

    Parameters
    ----------
    compiled:
        Whether the kernel compiled without errors.
    contracts_passed:
        Whether all implemented Kernel Contract gates passed.
    speedup_vs_handwritten:
        ``t_reference / t_candidate``; >1 means candidate is faster.
        Pass ``None`` if timing was not measured (scores the parity floor).
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
    log_speedup = math.log(speedup) if speedup > 0.0 else _SLOW_FLOOR_LOG
    shaped = min(max(log_speedup, _SLOW_FLOOR_LOG), _FAST_CAP_LOG)
    reward = _PARITY_ANCHOR + shaped

    if bug_routing and speedup > 1.0:
        reward += _BUG_ROUTING_BONUS

    return reward
