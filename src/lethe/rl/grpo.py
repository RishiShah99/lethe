"""GRPO trainer scaffolding.

This module defines the *interface* of the GRPO update step plus a
minimal trainer that wires the policy → rollout → reward → advantages
pipeline together.  The actual gradient update is encoded in
``compute_grpo_loss`` and ``compute_group_advantages`` below.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from lethe.rl.policy import PolicyInterface
from lethe.rl.reward import score_callable
from lethe.rl.rollout import Candidate, Rollout, ScoredCandidate


@dataclass(frozen=True)
class GRPOConfig:
    """Hyperparameters for one GRPO training step.

    Defaults track the DeepSeek-Math GRPO paper unless noted.
    """

    n_per_prompt: int = 8
    kl_coef: float = 0.04
    clip_eps: float = 0.2
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0


def compute_group_advantages(
    rewards: torch.Tensor,
    *,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Compute GRPO group-relative advantages for a single prompt's reward group.

    Parameters
    ----------
    rewards : torch.Tensor
        Shape ``(K,)`` scalar rewards for the *K* candidates sampled from
        one prompt.
    eps : float
        Numerical floor added to the standard deviation before division to
        prevent divide-by-zero on degenerate groups.  Default ``1e-4``
        matches practical GRPO implementations and is larger than the
        ``1e-8`` sometimes used in other normalisation contexts — the wider
        floor keeps gradients stable when rewards are nearly identical.

    Returns
    -------
    torch.Tensor
        Shape ``(K,)`` advantages, same dtype and device as *rewards*.

    Notes
    -----
    **Population std (unbiased=False).**
    The original DeepSeekMath GRPO paper normalises by the *population*
    standard deviation (dividing by K, not K-1).  Using the unbiased
    (Bessel-corrected) estimator would inflate advantages for small groups
    such as K=2 and produce different scaling than the reference
    implementation; population std is therefore used here.

    **Degenerate all-equal-rewards group.**
    When all rewards in the group are identical the population std is
    exactly zero, making every numerator ``r_i - mean(r) = 0``.  The
    result is the zero tensor regardless of *eps*.  This is the *intended*
    GRPO behaviour: an uninformative group (no relative signal between
    candidates) contributes zero policy gradient rather than a noisy
    signal inflated by a tiny denominator.
    """
    mean = rewards.mean()
    std = rewards.std(correction=0)
    return (rewards - mean) / (std + eps)


def compute_grpo_loss(
    advantages: torch.Tensor,
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    *,
    clip_eps: float,
    kl_coef: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """GRPO policy-gradient loss with PPO clipping and KL penalty.

    Implements the token-level GRPO objective from DeepSeekMath (Shao et
    al., 2024) with the Schulman k3 KL estimator for the reference-policy
    penalty.

    Parameters
    ----------
    advantages : torch.Tensor
        Shape ``(K,)`` group-relative advantage for each of the *K*
        sequences.  Typically produced by :func:`compute_group_advantages`.
    new_log_probs : torch.Tensor
        Shape ``(K, T)`` per-token log-probabilities under the *current*
        (trainable) policy.  Must carry ``requires_grad=True`` for
        ``loss.backward()`` to produce gradients.
    old_log_probs : torch.Tensor
        Shape ``(K, T)`` per-token log-probabilities under the *behaviour*
        (sampling) policy.  Detached inside this function — gradients must
        not flow through this argument.
    ref_log_probs : torch.Tensor
        Shape ``(K, T)`` per-token log-probabilities under the frozen
        reference policy.  Detached inside this function.
    clip_eps : float
        PPO probability-ratio clip radius.  Typical value: ``0.2``.
    kl_coef : float
        Coefficient on the KL penalty term.  Typical value: ``0.04``.
    mask : torch.Tensor or None
        Shape ``(K, T)`` boolean or float completion mask.  A value of
        ``True`` / ``1.0`` marks a *valid* token that should contribute to
        the loss; ``False`` / ``0.0`` marks a padding token.  When
        ``None`` all tokens are treated as valid.

    Returns
    -------
    torch.Tensor
        Scalar loss to minimise.  Gradient flows only through
        *new_log_probs*.

    Notes
    -----
    **Math (token level).**

    .. code-block:: text

        ratio_t        = exp(new_lp_t - old_lp_t)
        surrogate_t    = min(ratio_t * A,
                             clip(ratio_t, 1-clip_eps, 1+clip_eps) * A)
        kl_t           = exp(ref_lp_t - new_lp_t) - (ref_lp_t - new_lp_t) - 1
        obj_t          = surrogate_t - kl_coef * kl_t
        loss           = -mean_K( masked_mean_T(obj_t) )

    **PPO clip.**
    The standard two-sided min-clip is applied.  For positive advantages
    the surrogate is capped at ``(1 + clip_eps) * A``; for negative
    advantages it is capped at ``(1 - clip_eps) * A``.

    **Schulman k3 KL estimator.**
    ``kl_t = exp(d) - d - 1`` where ``d = ref_lp - new_lp`` is a
    non-negative, unbiased estimator of ``KL(new || ref)`` that is always
    ≥ 0.  It is zero iff ``new_lp == ref_lp`` at every token.

    **Masked mean.**
    The per-sequence token average divides by the *count of valid tokens*
    (clamped to a minimum of 1) so that short completions are not
    artificially down-weighted relative to long ones.
    """
    # Defensive detach — gradient must flow only through new_log_probs.
    old_lp = old_log_probs.detach()
    ref_lp = ref_log_probs.detach()

    # --- ratio and PPO surrogate ----------------------------------------
    # log_ratio shape: (K, T)
    log_ratio = new_log_probs - old_lp

    # ratio shape: (K, T)
    ratio = torch.exp(log_ratio)

    # advantages broadcast: (K,) -> (K, 1) for element-wise multiply with (K, T)
    adv = advantages.unsqueeze(1)

    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    surrogate = torch.minimum(unclipped, clipped)

    # --- Schulman k3 KL estimator ----------------------------------------
    # kl = exp(ref - new) - (ref - new) - 1  >=  0 always
    d = ref_lp - new_log_probs
    kl = torch.exp(d) - d - 1.0

    # --- per-token objective ---------------------------------------------
    per_token_obj = surrogate - kl_coef * kl

    # --- masked mean over tokens per sequence ----------------------------
    if mask is None:
        float_mask = torch.ones_like(per_token_obj)
    else:
        float_mask = mask.to(dtype=per_token_obj.dtype)

    token_sums = (per_token_obj * float_mask).sum(dim=1)  # (K,)
    token_counts = float_mask.sum(dim=1).clamp(min=1.0)  # (K,)
    seq_means = token_sums / token_counts  # (K,)

    # --- mean over sequences and negate for minimisation ----------------
    loss = -seq_means.mean()
    return loss


@dataclass
class StepMetrics:
    """Per-step metrics returned by ``GRPOTrainer.step``."""

    mean_reward: float
    max_reward: float
    n_compiled: int
    n_contracts_passed: int
    n_bug_routing: int
    loss: float | None = None  # None until compute_grpo_loss is implemented


class GRPOTrainer:
    """Coordinator for one GRPO update step.

    Owns the policy and reference policy, drives candidate generation,
    scoring, and the (future) gradient update. The actual training
    optimiser is supplied by the caller so this class stays
    framework-agnostic.
    """

    def __init__(
        self,
        config: GRPOConfig,
        policy: PolicyInterface,
        ref_policy: PolicyInterface,
        *,
        candidate_factory: Callable[[str, str, int, int], Candidate],
        compile_fn: Callable[[str], bool],
        import_fn: Callable[[Candidate], Callable[..., torch.Tensor]],
        reference_fn: Callable[..., torch.Tensor],
        benchmark_fn: Callable[[Callable[..., torch.Tensor]], float | None] | None = None,
        gate_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Construct a trainer.

        Parameters
        ----------
        config:
            Hyperparameters for the update step.
        policy:
            The policy being trained.
        ref_policy:
            Frozen reference policy used for the KL penalty.
        candidate_factory:
            ``(source, target_op, generation_id, candidate_idx) -> Candidate``.
            Wraps a raw generated string into a typed ``Candidate``.
        compile_fn:
            Boolean-returning compile hook (callers wire to
            ``lethe.verifier.compile.compile_kernel``).
        import_fn:
            Materialises a candidate into a callable. Wire to
            ``lethe.kernels.loader.import_candidate`` after
            writing the source to disk.
        reference_fn:
            The reference op the candidates are scored against.
        benchmark_fn:
            Optional timing hook; returns ``t_reference / t_candidate``
            (so >1 is "candidate is faster"). ``None`` skips speedup scoring.
        gate_kwargs:
            Per-gate overrides for shape / dtype / tolerance, forwarded to
            ``run_all_gates``.
        """
        self.config = config
        self.policy = policy
        self.ref_policy = ref_policy
        self._candidate_factory = candidate_factory
        self._compile_fn = compile_fn
        self._import_fn = import_fn
        self._reference_fn = reference_fn
        self._benchmark_fn = benchmark_fn
        self._gate_kwargs = gate_kwargs

    def rollout(
        self,
        prompt: str,
        target_op: str,
        *,
        prompt_id: str = "",
        generation_id: int = 0,
    ) -> Rollout:
        """Generate, compile, score, and bundle ``n_per_prompt`` candidates.

        Failed compiles short-circuit to a zero-reward ``ScoredCandidate``
        without running any contract gates.
        """
        sources = self.policy.generate(prompt, self.config.n_per_prompt)
        scored: list[ScoredCandidate] = []
        for idx, source in enumerate(sources):
            candidate = self._candidate_factory(source, target_op, generation_id, idx)
            compiled = self._compile_fn(candidate.source)
            if not compiled:
                scored.append(
                    score_callable(
                        candidate,
                        candidate_fn=lambda x: x,  # never called when compiled=False
                        reference_fn=self._reference_fn,
                        compiled=False,
                    )
                )
                continue

            candidate_fn = self._import_fn(candidate)
            speedup = self._benchmark_fn(candidate_fn) if self._benchmark_fn else None
            scored.append(
                score_callable(
                    candidate,
                    candidate_fn=candidate_fn,
                    reference_fn=self._reference_fn,
                    compiled=True,
                    speedup_vs_handwritten=speedup,
                    # set externally if the reference kernel trips
                    # CompileResult.blackwell_failure (C7907 or TMEM budget)
                    bug_routing=False,
                    gate_kwargs=self._gate_kwargs,
                )
            )
        return Rollout(prompt=prompt, prompt_id=prompt_id, candidates=tuple(scored))

    @staticmethod
    def summarise(rollout: Rollout) -> StepMetrics:
        """Compute summary metrics over a scored rollout."""
        if not rollout.candidates:
            return StepMetrics(
                mean_reward=0.0,
                max_reward=0.0,
                n_compiled=0,
                n_contracts_passed=0,
                n_bug_routing=0,
            )
        rewards = rollout.rewards
        return StepMetrics(
            mean_reward=sum(rewards) / len(rewards),
            max_reward=max(rewards),
            n_compiled=sum(1 for c in rollout.candidates if c.compiled),
            n_contracts_passed=sum(1 for c in rollout.candidates if c.contracts_passed),
            n_bug_routing=sum(1 for c in rollout.candidates if c.bug_routing),
        )
