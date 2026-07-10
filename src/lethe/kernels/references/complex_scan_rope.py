"""Mamba-3 real-equivalent SSM with data-dependent RoPE rotation (Eq. 9)."""

import math

import torch
from torch import Tensor


def _apply_rope_rotation(v: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply pairwise RoPE rotation to the last dimension of v."""
    num_rope = cos.shape[-1]
    rotary_dim = 2 * num_rope

    if rotary_dim > v.shape[-1]:
        raise ValueError(f"rotary_dim={rotary_dim} exceeds d_state={v.shape[-1]}")

    v_rot = v[..., :rotary_dim]  # (..., 2*num_rope)
    v_pass = v[..., rotary_dim:]  # (..., d_state - rotary_dim)  identity

    # reshape to (..., num_rope, 2)
    v_pairs = v_rot.reshape(*v_rot.shape[:-1], num_rope, 2)
    v0 = v_pairs[..., 0]  # even indices
    v1 = v_pairs[..., 1]  # odd indices

    # Standard 2-D rotation: [cos -sin; sin cos] @ [v0; v1]
    out0 = v0 * cos - v1 * sin
    out1 = v0 * sin + v1 * cos

    out_pairs = torch.stack([out0, out1], dim=-1)  # (..., num_rope, 2)
    out_rot = out_pairs.reshape(v_rot.shape)  # (..., 2*num_rope)

    return torch.cat([out_rot, v_pass], dim=-1)


def reference_complex_scan_rope(
    x: Tensor,
    B: Tensor,
    C: Tensor,
    dt: Tensor,
    A: Tensor,
    angle_proj: Tensor,
) -> Tensor:
    """Real-equivalent SSM with data-dependent RoPE rotation (Mamba-3 §3.2)."""
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"Expected float32/float64 input, got {x.dtype}")

    batch, seqlen, nheads, headdim = x.shape
    d_state = B.shape[-1]
    num_rope = angle_proj.shape[-1]

    delta_angle = torch.tanh(angle_proj) * dt.unsqueeze(-1) * math.pi
    # cumulative sum along sequence dim (causal)
    theta = torch.cumsum(delta_angle, dim=1)  # (B, L, H, num_rope)
    theta = torch.remainder(theta, 2.0 * math.pi)  # numerical stability

    # Pre-compute cos/sin of cumulative angles: (B, L, H, num_rope)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    B_flat = B.reshape(batch * seqlen * nheads, d_state)
    C_flat = C.reshape(batch * seqlen * nheads, d_state)
    cos_flat = cos_theta.reshape(batch * seqlen * nheads, num_rope)
    sin_flat = sin_theta.reshape(batch * seqlen * nheads, num_rope)

    B_rot_flat = _apply_rope_rotation(B_flat, cos_flat, sin_flat)
    C_rot_flat = _apply_rope_rotation(C_flat, cos_flat, sin_flat)

    B_rot = B_rot_flat.reshape(batch, seqlen, nheads, d_state)
    C_rot = C_rot_flat.reshape(batch, seqlen, nheads, d_state)

    alpha = torch.exp(dt * A.unsqueeze(0).unsqueeze(0))  # (B, L, H)

    h = torch.zeros(batch, nheads, headdim, d_state, dtype=x.dtype, device=x.device)
    y = torch.empty_like(x)

    for t in range(seqlen):
        # alpha_t: (batch, nheads) -> (batch, nheads, 1, 1)
        alpha_t = alpha[:, t, :].unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)

        # dt_t: (batch, nheads) -> (batch, nheads, 1, 1)
        dt_t = dt[:, t, :].unsqueeze(-1).unsqueeze(-1)

        # B_rot_t: (batch, nheads, d_state) -> (batch, nheads, 1, d_state)
        B_rot_t = B_rot[:, t, :, :].unsqueeze(2)

        # x_t: (batch, nheads, headdim) -> (batch, nheads, headdim, 1)
        x_t = x[:, t, :, :].unsqueeze(-1)  # (B, H, P, 1)

        # h_t = alpha_t * h_{t-1} + dt_t * B_rot_t * x_t, broadcast over (B, H, P, N)
        h = alpha_t * h + dt_t * B_rot_t * x_t  # (B, H, P, N)

        # --- readout: y_t = C_rot_t^T @ h_t ---
        C_rot_t = C_rot[:, t, :, :].unsqueeze(2)  # (B, H, 1, N)
        # (h * C_rot_t).sum(-1): (B, H, P, N) -> (B, H, P)
        y_t = (h * C_rot_t).sum(-1)  # (B, H, P)
        # store: y is (batch, seqlen, nheads, headdim)
        y[:, t, :, :] = y_t  # (B, H, P) -> y[:,t,:,:]

    return y
