"""CPU replica of the C4 Triton kernel's exact algorithm vs the reference.

The kernel computes the same math as the reference through a different
path: theta as a per-step-remaindered running sum duplicated across each
pair's two lanes (reference: one mod after a full cumsum over k-indexed
angles), the rotation as self*cos + sign*partner*sin per lane (reference:
pair-reshape matrix form), fp32 throughout. This replica mirrors the
kernel statement-for-statement so any algebra divergence — sign flip,
partner mapping, identity-tail boundary, remainder placement — surfaces
on CPU without a GPU in the loop. Tolerances are fp32 reorder noise, not
correctness slack.
"""

from __future__ import annotations

import math

import torch

from flash_mamba_rl.kernels.references.complex_scan_rope import reference_complex_scan_rope


def _kernel_replica(
    x: torch.Tensor,
    b_in: torch.Tensor,
    c_in: torch.Tensor,
    dt: torch.Tensor,
    a_head: torch.Tensor,
    angle_proj: torch.Tensor,
) -> torch.Tensor:
    """The Triton kernel's compute order, statement for statement, in torch."""
    batch, seq_len, nheads, headdim = x.shape
    n_state = b_in.shape[-1]
    s_angles = angle_proj.shape[-1]
    two_pi = 6.283185307179586

    offs_n = torch.arange(n_state)
    pair_k = offs_n // 2
    mask_rot = pair_k < s_angles
    # The kernel masks the partner load (partner < n_state, other=0) and the
    # value is only consumed on rotated lanes, whose partner is always
    # in-range; torch gather needs the explicit clamp instead of a mask.
    partner = (offs_n ^ 1).clamp(max=n_state - 1)
    sign = torch.where(offs_n % 2 == 0, -1.0, 1.0)

    y = torch.empty_like(x)
    h = torch.zeros(batch, nheads, headdim, n_state)
    theta = torch.zeros(batch, nheads, n_state)

    ang_lane = torch.where(
        mask_rot, angle_proj[:, :, :, pair_k.clamp(max=max(s_angles - 1, 0))], 0.0
    )
    for t in range(seq_len):
        dt_t = dt[:, t].unsqueeze(-1)
        theta = theta + torch.where(mask_rot, torch.tanh(ang_lane[:, t]) * dt_t * math.pi, 0.0)
        theta = theta - torch.floor(theta / two_pi) * two_pi
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        b_self = b_in[:, t]
        b_part = b_in[:, t, :, partner]
        c_self = c_in[:, t]
        c_part = c_in[:, t, :, partner]
        b_rot = torch.where(mask_rot, b_self * cos_t + sign * b_part * sin_t, b_self)
        c_rot = torch.where(mask_rot, c_self * cos_t + sign * c_part * sin_t, c_self)

        alpha_t = torch.exp(dt[:, t] * a_head)
        bu = (dt[:, t].unsqueeze(-1) * b_rot).unsqueeze(2) * x[:, t].unsqueeze(-1)
        h = alpha_t.unsqueeze(-1).unsqueeze(-1) * h + bu
        y[:, t] = (h * c_rot.unsqueeze(2)).sum(-1)
    return y


class TestKernelReplicaParity:
    def _check(self, b: int, seq: int, h: int, p: int, n: int, s: int, seed: int = 0) -> None:
        torch.manual_seed(seed)
        x = torch.randn(b, seq, h, p)
        bb = torch.randn(b, seq, h, n)
        cc = torch.randn(b, seq, h, n)
        dt = torch.rand(b, seq, h) * 0.1 + 1e-3
        a = -torch.rand(h)
        angle = torch.randn(b, seq, h, s)
        got = _kernel_replica(x, bb, cc, dt, a, angle)
        want = reference_complex_scan_rope(x, bb, cc, dt, a, angle)
        max_err = (got - want).abs().max().item()
        scale = want.abs().max().clamp(min=1.0).item()
        assert max_err / scale < 1e-5, f"replica diverges: scale_rel={max_err / scale:.3e}"

    def test_partial_rotary(self) -> None:
        self._check(2, 64, 2, 4, 16, 6)

    def test_full_rotary(self) -> None:
        self._check(2, 64, 3, 5, 16, 8)

    def test_odd_n_state_unrotated_tail(self) -> None:
        self._check(2, 48, 2, 4, 11, 4)

    def test_single_pair(self) -> None:
        self._check(1, 32, 1, 1, 2, 1)

    def test_l1(self) -> None:
        self._check(1, 1, 2, 4, 8, 3)

    def test_longer_sequence_angle_wraps(self) -> None:
        # Long enough that theta wraps 2*pi several times: the per-step
        # remainder path and the reference's mod-after-cumsum must agree
        # through the wraps.
        self._check(1, 512, 2, 3, 8, 4, seed=7)

    def test_fp64_tight(self) -> None:
        torch.manual_seed(11)
        b, seq, h, p, n, s = 2, 96, 2, 4, 16, 6
        x = torch.randn(b, seq, h, p, dtype=torch.float64)
        bb = torch.randn(b, seq, h, n, dtype=torch.float64)
        cc = torch.randn(b, seq, h, n, dtype=torch.float64)
        dt = (torch.rand(b, seq, h) * 0.1 + 1e-3).to(torch.float64)
        a = -torch.rand(h).to(torch.float64)
        angle = torch.randn(b, seq, h, s, dtype=torch.float64)
        got = _kernel_replica(x, bb, cc, dt, a, angle)
        want = reference_complex_scan_rope(x, bb, cc, dt, a, angle)
        max_err = (got - want).abs().max().item()
        assert max_err < 1e-12, f"fp64 replica diverges: {max_err:.3e}"
