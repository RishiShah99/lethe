"""Mamba-3 real-equivalent SSM scan with data-dependent RoPE (eager PyTorch)."""

import math

import torch
from torch import Tensor


def _rotate_pairs(v: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    num_rope = cos.shape[-1]
    rotary_dim = 2 * num_rope

    v_rot = v[..., :rotary_dim]
    v_pass = v[..., rotary_dim:]

    v_pairs = v_rot.reshape(*v_rot.shape[:-1], num_rope, 2)
    v0 = v_pairs[..., 0]
    v1 = v_pairs[..., 1]

    out0 = v0 * cos - v1 * sin
    out1 = v0 * sin + v1 * cos

    out_pairs = torch.stack([out0, out1], dim=-1)
    out_rot = out_pairs.reshape(v_rot.shape)
    return torch.cat([out_rot, v_pass], dim=-1)


def complex_scan_rope(
    x: Tensor,
    B: Tensor,
    C: Tensor,
    dt: Tensor,
    A: Tensor,
    angle_proj: Tensor,
) -> Tensor:
    out_dtype = x.dtype
    if out_dtype in (torch.float16, torch.bfloat16):
        x, B, C, dt, A, angle_proj = (t.to(torch.float32) for t in (x, B, C, dt, A, angle_proj))

    batch, seqlen, nheads, headdim = x.shape
    d_state = B.shape[-1]
    num_rope = angle_proj.shape[-1]

    delta_angle = torch.tanh(angle_proj) * dt.unsqueeze(-1) * math.pi
    theta = torch.cumsum(delta_angle, dim=1)
    theta = torch.remainder(theta, 2.0 * math.pi)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    B_flat = B.reshape(batch * seqlen * nheads, d_state)
    C_flat = C.reshape(batch * seqlen * nheads, d_state)
    cos_flat = cos_theta.reshape(batch * seqlen * nheads, num_rope)
    sin_flat = sin_theta.reshape(batch * seqlen * nheads, num_rope)

    B_rot = _rotate_pairs(B_flat, cos_flat, sin_flat).reshape(batch, seqlen, nheads, d_state)
    C_rot = _rotate_pairs(C_flat, cos_flat, sin_flat).reshape(batch, seqlen, nheads, d_state)

    alpha = torch.exp(dt * A.unsqueeze(0).unsqueeze(0))

    h = torch.zeros(batch, nheads, headdim, d_state, dtype=x.dtype, device=x.device)
    y = torch.empty_like(x)

    for t in range(seqlen):
        alpha_t = alpha[:, t, :].unsqueeze(-1).unsqueeze(-1)
        dt_t = dt[:, t, :].unsqueeze(-1).unsqueeze(-1)
        B_rot_t = B_rot[:, t, :, :].unsqueeze(2)
        x_t = x[:, t, :, :].unsqueeze(-1)

        h = alpha_t * h + dt_t * B_rot_t * x_t

        C_rot_t = C_rot[:, t, :, :].unsqueeze(2)
        y[:, t, :, :] = (h * C_rot_t).sum(-1)

    return y.to(out_dtype)
