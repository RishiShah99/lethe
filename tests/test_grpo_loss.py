"""Exhaustive unit tests for compute_group_advantages and compute_grpo_loss."""

from __future__ import annotations

import math

import pytest
import torch

from lethe.rl.grpo import compute_group_advantages, compute_grpo_loss


def _lp(data: list[list[float]], *, requires_grad: bool = False) -> torch.Tensor:
    """Build a (K, T) log-prob tensor."""
    t = torch.tensor(data, dtype=torch.float64)
    if requires_grad:
        t.requires_grad_(True)
    return t


def _adv(data: list[float]) -> torch.Tensor:
    return torch.tensor(data, dtype=torch.float64)


class TestComputeGroupAdvantages:
    def test_hand_computed_4_element(self) -> None:
        # hand-computed: mean=2.5, population std=sqrt(1.25), A_i=(r_i-2.5)/(sqrt(1.25)+eps).
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        eps = 1e-4
        std = rewards.std(unbiased=False).item()
        mean = rewards.mean().item()
        expected = [(r - mean) / (std + eps) for r in [1.0, 2.0, 3.0, 4.0]]

        out = compute_group_advantages(rewards, eps=eps)

        assert out.shape == (4,)
        for i, exp in enumerate(expected):
            assert math.isclose(out[i].item(), exp, rel_tol=1e-9), (
                f"element {i}: got {out[i].item()}, expected {exp}"
            )

    def test_all_equal_rewards_returns_zeros(self) -> None:
        # All rewards identical: numerator = r_i - mean = 0 for every i.
        rewards = torch.tensor([3.0, 3.0, 3.0, 3.0], dtype=torch.float64)
        out = compute_group_advantages(rewards)
        assert out.shape == (4,)
        assert torch.all(out == 0.0), f"expected zeros, got {out}"

    def test_random_group_mean_near_zero_and_std_near_one(self) -> None:
        # for large K, normalised advantages should have mean ≈ 0 and variance ≈ 1.
        torch.manual_seed(42)
        rewards = torch.randn(64, dtype=torch.float64)
        eps = 1e-4
        out = compute_group_advantages(rewards, eps=eps)
        assert math.isclose(out.mean().item(), 0.0, abs_tol=1e-6), (
            f"mean not ~0: {out.mean().item()}"
        )
        # Population variance of out ≈ (std / (std + eps))^2 ≈ 1 for large std.
        pop_var = (out - out.mean()).pow(2).mean().item()
        assert math.isclose(pop_var, 1.0, rel_tol=0.01), f"population variance not ~1: {pop_var}"

    def test_shape_preserved(self) -> None:
        for k in (1, 2, 8, 16):
            rewards = torch.randn(k)
            out = compute_group_advantages(rewards)
            assert out.shape == (k,), f"K={k}: shape {out.shape}"

    def test_single_element_group(self) -> None:
        # K=1: mean=r, std=0, output = 0 / eps = 0.
        rewards = torch.tensor([5.0])
        out = compute_group_advantages(rewards)
        assert out.shape == (1,)
        assert out[0].item() == pytest.approx(0.0, abs=1e-9)


class TestComputeGrpoLossHandComputed:
    """Hand-verified K=2,T=2 case: ratio=1 everywhere, clip_eps=0.2, kl_coef=1.0, no mask."""

    def test_hand_computed_value(self) -> None:
        adv = _adv([1.0, -1.0])
        new_lp = _lp([[-1.0, -2.0], [-1.0, -2.0]], requires_grad=True)
        old_lp = _lp([[-1.0, -2.0], [-1.0, -2.0]])
        ref_lp = _lp([[-2.0, -1.0], [-2.0, -1.0]])

        loss = compute_grpo_loss(
            adv,
            new_lp,
            old_lp,
            ref_lp,
            clip_eps=0.2,
            kl_coef=1.0,
        )

        # expected = cosh(1) - 1
        expected = math.cosh(1.0) - 1.0
        assert math.isclose(loss.item(), expected, rel_tol=1e-6), (
            f"got {loss.item()}, expected {expected} (cosh(1)-1)"
        )


class TestComputeGrpoLossIdenticalPolicies:
    """new == old == ref: ratio=1, kl=0 => loss = -mean(advantages)."""

    def test_identical_policies(self) -> None:
        # ratio=1, no clip; kl=0; surrogate=A; token mean = A; loss = -mean(A)
        K, T = 4, 5
        adv = _adv([1.0, -0.5, 0.25, -0.75])
        lp = torch.full((K, T), -1.0, dtype=torch.float64)
        new_lp = lp.clone().requires_grad_(True)

        loss = compute_grpo_loss(adv, new_lp, lp, lp, clip_eps=0.2, kl_coef=0.04)

        expected = -adv.mean().item()
        assert math.isclose(loss.item(), expected, rel_tol=1e-9), (
            f"got {loss.item()}, expected {expected}"
        )


class TestComputeGrpoLossClipping:
    """Clipping tests: ratio outside [1-eps, 1+eps] is truncated."""

    def test_clip_upper_positive_advantage(self) -> None:
        # ratio >> 1+eps, A > 0 => clipped to (1+eps)*A.
        clip_eps = 0.2
        adv = _adv([1.0])
        old_lp = _lp([[-5.0]])  # new - old = 5 => ratio = exp(5) >> 1.2
        ref_lp = _lp([[-5.0]])

        new_lp_a = _lp([[-0.0]], requires_grad=True)  # ratio = exp(5) >> 1.2
        new_lp_b = _lp([[1.0]], requires_grad=True)  # ratio even larger

        loss_a = compute_grpo_loss(adv, new_lp_a, old_lp, ref_lp, clip_eps=clip_eps, kl_coef=0.0)
        loss_b = compute_grpo_loss(adv, new_lp_b, old_lp, ref_lp, clip_eps=clip_eps, kl_coef=0.0)

        # Both should equal -(1+eps)*A = -1.2
        assert math.isclose(loss_a.item(), -1.2, rel_tol=1e-6), (
            f"loss_a={loss_a.item()}, expected -1.2"
        )
        assert math.isclose(loss_b.item(), loss_a.item(), rel_tol=1e-9), (
            "further new_lp increase changed loss past clip"
        )

    def test_clip_lower_negative_advantage(self) -> None:
        # ratio << 1-eps, A < 0: clip = (1-eps)*A = 0.8*(-1) = -0.8, the min-selected branch.
        clip_eps = 0.2
        adv = _adv([-1.0])
        # old_lp = 0, new_lp = log(0.1) + 0 = log(0.1) => ratio = 0.1
        new_lp_a = _lp([[math.log(0.1)]], requires_grad=True)
        # new_lp even lower => ratio even smaller => unclipped even closer to 0
        new_lp_b = _lp([[math.log(0.01)]], requires_grad=True)
        old_lp = _lp([[0.0]])
        ref_lp = _lp([[0.0]])

        loss_a = compute_grpo_loss(adv, new_lp_a, old_lp, ref_lp, clip_eps=clip_eps, kl_coef=0.0)
        loss_b = compute_grpo_loss(adv, new_lp_b, old_lp, ref_lp, clip_eps=clip_eps, kl_coef=0.0)

        # Both clipped at (1-eps)*A = 0.8*(-1) = -0.8 => loss = -(-0.8) = 0.8
        assert math.isclose(loss_a.item(), 0.8, rel_tol=1e-6), (
            f"loss_a={loss_a.item()}, expected 0.8"
        )
        assert math.isclose(loss_b.item(), 0.8, rel_tol=1e-6), (
            f"loss_b={loss_b.item()}, expected 0.8"
        )


class TestComputeGrpoLossKL:
    """Tests for the Schulman k3 KL estimator."""

    def test_kl_nonnegative_random(self) -> None:
        # kl = exp(d) - d - 1 >= 0 for all real d (equality iff d=0).
        torch.manual_seed(0)
        K, T = 8, 16
        adv = torch.zeros(K, dtype=torch.float64)
        new_lp = torch.randn(K, T, dtype=torch.float64, requires_grad=True)
        old_lp = torch.zeros(K, T, dtype=torch.float64)
        ref_lp = torch.randn(K, T, dtype=torch.float64)

        loss = compute_grpo_loss(adv, new_lp, old_lp, ref_lp, clip_eps=0.2, kl_coef=1.0)
        # loss = -mean_K(mean_T(0 - 1.0 * kl_t)) = +mean(kl) >= 0
        assert loss.item() >= 0.0, f"kl-only loss is negative: {loss.item()}"

    def test_kl_zero_iff_new_equals_ref(self) -> None:
        # When new_lp == ref_lp: d=0, exp(0)-0-1=0; kl term vanishes entirely.
        K, T = 4, 8
        lp = torch.randn(K, T, dtype=torch.float64)
        new_lp = lp.clone().requires_grad_(True)
        adv = torch.zeros(K, dtype=torch.float64)
        old_lp = lp.clone()

        loss = compute_grpo_loss(adv, new_lp, old_lp, lp, clip_eps=0.2, kl_coef=1.0)
        # advantages=0, kl=0 => loss = 0
        assert loss.item() == pytest.approx(0.0, abs=1e-12), (
            f"kl not zero when new==ref: loss={loss.item()}"
        )

    def test_zero_advantages_loss_reduces_kl(self) -> None:
        # With advantages=0 the surrogate is 0; loss = +kl_coef * mean(kl).
        K, T = 2, 4
        torch.manual_seed(7)
        adv = torch.zeros(K, dtype=torch.float64)
        ref_lp = torch.zeros(K, T, dtype=torch.float64)
        old_lp = torch.zeros(K, T, dtype=torch.float64)
        new_lp = torch.randn(K, T, dtype=torch.float64).requires_grad_(True)

        loss = compute_grpo_loss(adv, new_lp, old_lp, ref_lp, clip_eps=0.2, kl_coef=1.0)
        loss.backward()

        # After one gradient step (gradient descent), new_lp moves toward ref_lp=0.
        assert new_lp.grad is not None
        new_lp_after = new_lp - 0.1 * new_lp.grad

        # KL after step should be smaller.
        d_before = ref_lp - new_lp.detach()
        kl_before = (torch.exp(d_before) - d_before - 1.0).mean().item()
        d_after = ref_lp - new_lp_after.detach()
        kl_after = (torch.exp(d_after) - d_after - 1.0).mean().item()
        assert kl_after < kl_before, f"KL did not decrease: before={kl_before}, after={kl_after}"


class TestComputeGrpoLossGradients:
    """Gradient flow and direction tests."""

    def test_gradient_flows_through_new_only(self) -> None:
        K, T = 3, 4
        torch.manual_seed(1)
        adv = torch.randn(K, dtype=torch.float64)
        new_lp = torch.randn(K, T, dtype=torch.float64, requires_grad=True)
        old_lp = torch.randn(K, T, dtype=torch.float64, requires_grad=True)
        ref_lp = torch.randn(K, T, dtype=torch.float64, requires_grad=True)

        loss = compute_grpo_loss(adv, new_lp, old_lp, ref_lp, clip_eps=0.2, kl_coef=0.04)
        loss.backward()

        assert new_lp.grad is not None, "new_lp has no gradient"
        assert old_lp.grad is None, "old_lp should have no gradient (detached)"
        assert ref_lp.grad is None, "ref_lp should have no gradient (detached)"

    def test_gradient_direction_positive_advantage(self) -> None:
        # Single token, positive advantage, no KL (kl_coef=0).
        adv = _adv([1.0])
        new_lp = _lp([[-1.0]], requires_grad=True)
        old_lp = _lp([[-1.0]])  # ratio=1, no clip
        ref_lp = _lp([[-1.0]])

        loss = compute_grpo_loss(adv, new_lp, old_lp, ref_lp, clip_eps=0.2, kl_coef=0.0)
        loss.backward()

        assert new_lp.grad is not None
        # Gradient must be negative: minimising loss pushes new_lp up (toward higher prob).
        assert new_lp.grad[0, 0].item() < 0.0, (
            f"Expected grad < 0 for positive advantage; got {new_lp.grad[0, 0].item()}"
        )


class TestComputeGrpoLossMask:
    """Mask tests: padded tokens must be excluded from the loss."""

    def test_mask_half_tokens(self) -> None:
        # K=2, T=4: first 2 tokens valid, last 2 padded.
        K, T = 2, 4
        torch.manual_seed(3)
        adv = _adv([1.0, -1.0])

        # Build log-probs for the full T=4 sequence.
        new_full = torch.randn(K, T, dtype=torch.float64, requires_grad=True)
        old_full = torch.randn(K, T, dtype=torch.float64)
        ref_full = torch.randn(K, T, dtype=torch.float64)

        # Mask: only first 2 tokens are valid.
        mask = torch.zeros(K, T, dtype=torch.float64)
        mask[:, :2] = 1.0

        loss_masked = compute_grpo_loss(
            adv,
            new_full,
            old_full,
            ref_full,
            clip_eps=0.2,
            kl_coef=0.04,
            mask=mask,
        )

        # Compute loss on the T=2 prefix tensors directly (no mask).
        new_prefix = new_full[:, :2].detach().requires_grad_(True)
        old_prefix = old_full[:, :2]
        ref_prefix = ref_full[:, :2]

        loss_prefix = compute_grpo_loss(
            adv,
            new_prefix,
            old_prefix,
            ref_prefix,
            clip_eps=0.2,
            kl_coef=0.04,
        )

        assert math.isclose(loss_masked.item(), loss_prefix.item(), rel_tol=1e-9), (
            f"masked loss {loss_masked.item()} != prefix loss {loss_prefix.item()}"
        )

    def test_mask_none_equals_all_ones_mask(self) -> None:
        K, T = 3, 5
        torch.manual_seed(5)
        adv = torch.randn(K, dtype=torch.float64)
        new_lp = torch.randn(K, T, dtype=torch.float64, requires_grad=True)
        old_lp = torch.randn(K, T, dtype=torch.float64)
        ref_lp = torch.randn(K, T, dtype=torch.float64)
        all_ones = torch.ones(K, T, dtype=torch.float64)

        loss_no_mask = compute_grpo_loss(adv, new_lp, old_lp, ref_lp, clip_eps=0.2, kl_coef=0.04)
        loss_all_mask = compute_grpo_loss(
            adv, new_lp, old_lp, ref_lp, clip_eps=0.2, kl_coef=0.04, mask=all_ones
        )
        assert math.isclose(loss_no_mask.item(), loss_all_mask.item(), rel_tol=1e-12)

    def test_single_token_per_sequence_via_mask(self) -> None:
        # K=2, T=3 but only token 0 is valid per sequence.
        K, T = 2, 3
        adv = _adv([2.0, -2.0])
        # ratio=1 everywhere, kl=0.
        lp = torch.full((K, T), -1.0, dtype=torch.float64)
        new_lp = lp.clone().requires_grad_(True)
        mask = torch.zeros(K, T, dtype=torch.float64)
        mask[:, 0] = 1.0

        loss = compute_grpo_loss(adv, new_lp, lp, lp, clip_eps=0.2, kl_coef=0.0, mask=mask)

        # surrogate = A, kl = 0; masked mean = A (1 token each); loss = -mean([2,-2]) = 0.
        assert loss.item() == pytest.approx(0.0, abs=1e-12)


class TestComputeGrpoLossOutput:
    """Output shape and scalar invariants."""

    def test_output_is_scalar(self) -> None:
        K, T = 5, 10
        adv = torch.randn(K, dtype=torch.float64)
        lp = torch.randn(K, T, dtype=torch.float64)
        new_lp = lp.clone().requires_grad_(True)
        loss = compute_grpo_loss(adv, new_lp, lp, lp, clip_eps=0.2, kl_coef=0.04)
        assert loss.shape == ()

    def test_zero_advantages_zero_kl_produces_zero_loss(self) -> None:
        # adv=0, old==new==ref => surrogate=0, kl=0 => loss=0.
        K, T = 4, 6
        adv = torch.zeros(K, dtype=torch.float64)
        lp = torch.randn(K, T, dtype=torch.float64)
        new_lp = lp.clone().requires_grad_(True)
        loss = compute_grpo_loss(adv, new_lp, lp, lp, clip_eps=0.2, kl_coef=0.04)
        assert loss.item() == pytest.approx(0.0, abs=1e-12)

    def test_kl_coef_zero_recovers_ppo(self) -> None:
        # With kl_coef=0: loss is pure PPO surrogate, independent of ref_log_probs.
        K, T = 2, 3
        torch.manual_seed(9)
        adv = torch.randn(K, dtype=torch.float64)
        new_lp = torch.randn(K, T, dtype=torch.float64, requires_grad=True)
        old_lp = torch.randn(K, T, dtype=torch.float64)
        ref_a = torch.randn(K, T, dtype=torch.float64)
        ref_b = torch.randn(K, T, dtype=torch.float64)

        loss_a = compute_grpo_loss(adv, new_lp, old_lp, ref_a, clip_eps=0.2, kl_coef=0.0)
        loss_b = compute_grpo_loss(adv, new_lp, old_lp, ref_b, clip_eps=0.2, kl_coef=0.0)

        assert math.isclose(loss_a.item(), loss_b.item(), rel_tol=1e-12), (
            "kl_coef=0 loss should not depend on ref_log_probs"
        )
