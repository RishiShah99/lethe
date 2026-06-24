"""Mamba-3 real-equivalent SSM with data-dependent RoPE rotation (Eq. 9).

Implements the complex SSM reformulation from:
  "Mamba-3" (Lahoti, Li, et al., ICLR 2026, arXiv:2603.15569)
  Section 3.2, Equations 8-9 and Proposition 3.2.1.

SIGNATURE CHANGES FROM OLD STUB
---------------------------------
The old stub accepted (u, real, imag, theta, decay, B_proj, C_proj, D) and
raised NotImplementedError.  That signature predated the math resolution and
modelled an incorrect abstraction (separate real/imag projections and a fixed
theta schedule).  The resolved math shows:

  * B and C are single real vectors of size d_state carrying both the real and
    imaginary parts in interleaved pairs — there is no separate B_hat / C_hat
    projection (Prop 3.2.1; one d_state-wide linear layer, pairs (2k, 2k+1)).
  * theta is DATA-DEPENDENT: accumulated as a causal cumsum over
    tanh(angle_proj) * dt * pi, not a fixed frequency schedule (Q2b; C6).
  * The discretisation parameter alpha = exp(dt * A) is now explicit (dt and A
    enter as separate scalars, matching Eq. 9).
  * The old D skip-connection is out of scope for this oracle (the kernel
    verifier grades the SSM kernel only; D is handled by the outer Mamba block).

ORACLE SCOPE
------------
Implements the official rotary-kernel formulation (C2/C3): the cumulative
rotation is folded into B and C as a preprocessing step, and the scan itself
is a plain decay scan —

    B_rot_t = R(Theta_t) @ B_t,   C_rot_t = R(Theta_t) @ C_t
    h_t = exp(dt_t * A_t) * h_{t-1} + dt_t * B_rot_t * x_t
    y_t = C_rot_t^T @ h_t

where Theta_t is the CUMULATIVE angle and R the block-diagonal pairwise
rotation.  This is equivalent to Eq. 9's per-step state rotation
(h_t = alpha_t * R(delta_theta_t) @ h_{t-1} + ...) by the change of basis
h_hat_t = R(-Theta_t) @ h_t, up to the sign of angle_proj (a learnable
reparameterisation).  The official kernels use the fold-into-B/C form — the
state is never re-rotated inside the scan — so the oracle matches that
convention exactly.

The trapezoidal discretisation (Prop 3.2.2) reduces to this form at lambda=1;
this oracle implements the lambda=1 base case.

UNRESOLVED / OUT OF SCOPE
--------------------------
  * B_bias / C_bias (shape (nheads, mimo_rank, d_state), initialised to 1):
    observed in C1 mamba3.py but their role is not stated in the main-paper
    text.  Not included here.
  * The trapezoidal lambda term (beta/gamma split of Prop 3.2.2): the oracle
    follows Eq. 9 only.
  * D skip connection: handled by the outer block, not the SSM scan.
"""

import math

import torch
from torch import Tensor


def _apply_rope_rotation(v: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply pairwise RoPE rotation to the last dimension of v.

    Operates on consecutive pairs (2k, 2k+1) of the last dimension.
    Beyond 2*num_rope_angles the dimensions are left unchanged (identity).

    conjugate=False throughout (same rotation sign for B and C); confirmed
    from C3 apply_rotary_qk_inference_reference called without conjugation.

    Args:
        v:   Tensor of shape (..., d_state), float32.
        cos: Tensor of shape (..., num_rope_angles), float32.
        sin: Tensor of shape (..., num_rope_angles), float32.

    Returns:
        Rotated tensor, same shape as v.
    """
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
    """Real-equivalent SSM with data-dependent RoPE rotation (Mamba-3 §3.2).

    Official rotary-kernel formulation (C2/C3) — rotation folded into B/C,
    plain decay scan:
        B_rot_t = R(Theta_t) @ B_t,   C_rot_t = R(Theta_t) @ C_t
        h_t = alpha_t * h_{t-1} + dt_t * B_rot_t * x_t
        y_t = C_rot_t^T @ h_t

    where:
        alpha_t = exp(dt_t * A_t)       — scalar per (batch, head, step)
        R       = block-diag pairwise rotation
        Theta_t = cumsum_s<=t[ tanh(angle_proj_s) * dt_s * pi ]  mod 2*pi

    The hidden state is NEVER re-rotated inside the scan — the cumulative
    rotation lives entirely in B_rot/C_rot.  Equivalent to Eq. 9's per-step
    state rotation via the change of basis h_hat_t = R(-Theta_t) @ h_t (up to
    the sign of angle_proj); see the module docstring.

    Angle accumulation is computed as a preprocessing step before the scan
    loop (structural requirement; confirmed from C2/C6 where angle_dt_fwd()
    is called before the SSM kernel).  Modulo 2*pi is applied for numerical
    stability (C6).

    B and C encode both real and imaginary parts in interleaved pairs
    (2k, 2k+1) — no separate B_hat / C_hat projection exists (Prop 3.2.1).
    Pairwise rotation sign is the same for B and C (conjugate=False, C3).
    Dimensions beyond 2*num_rope_angles are identity-rotated.

    Shape contracts
    ---------------
    x          : (batch, seqlen, nheads, headdim)         float32
    B          : (batch, seqlen, nheads, d_state)          float32
    C          : (batch, seqlen, nheads, d_state)          float32
    dt         : (batch, seqlen, nheads)                   float32, positive
    A          : (nheads,)                                 float32, negative
    angle_proj : (batch, seqlen, nheads, num_rope_angles)  float32

    Returns
    -------
    y : (batch, seqlen, nheads, headdim)  float32

    Notes
    -----
    * Scan loop is an explicit Python for-loop over t (oracle, not kernel).
    * No D skip, no B/C bias, no trapezoidal lambda.  These belong to the
      outer Mamba-3 block, not the SSM scan kernel being verified.
    * The trapezoidal discretisation (Prop 3.2.2) reduces to this at lambda=1.
    * float64 is accepted alongside float32 so gradcheck-style tests can
      validate this exact function (half precision stays rejected).
    """
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"Expected float32/float64 input, got {x.dtype}")

    batch, seqlen, nheads, headdim = x.shape
    d_state = B.shape[-1]
    num_rope = angle_proj.shape[-1]

    # -----------------------------------------------------------------
    # Step 1: Angle accumulation (preprocessing, before scan).
    # Theta_t = cumsum_s<=t[ tanh(angle_proj_s) * dt_s * pi ]  mod 2*pi
    # angle_proj: (B, L, H, S);  dt: (B, L, H)
    # -----------------------------------------------------------------
    delta_angle = torch.tanh(angle_proj) * dt.unsqueeze(-1) * math.pi
    # cumulative sum along sequence dim (causal)
    theta = torch.cumsum(delta_angle, dim=1)  # (B, L, H, num_rope)
    theta = torch.remainder(theta, 2.0 * math.pi)  # numerical stability

    # Pre-compute cos/sin of cumulative angles: (B, L, H, num_rope)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    # -----------------------------------------------------------------
    # Step 2: Apply RoPE rotation to B and C (same sign, conjugate=False)
    # B_rot: (B, L, H, d_state);  C_rot: same
    # -----------------------------------------------------------------
    # Flatten batch/L/H for vectorised rotation
    B_flat = B.reshape(batch * seqlen * nheads, d_state)
    C_flat = C.reshape(batch * seqlen * nheads, d_state)
    cos_flat = cos_theta.reshape(batch * seqlen * nheads, num_rope)
    sin_flat = sin_theta.reshape(batch * seqlen * nheads, num_rope)

    B_rot_flat = _apply_rope_rotation(B_flat, cos_flat, sin_flat)
    C_rot_flat = _apply_rope_rotation(C_flat, cos_flat, sin_flat)

    B_rot = B_rot_flat.reshape(batch, seqlen, nheads, d_state)
    C_rot = C_rot_flat.reshape(batch, seqlen, nheads, d_state)

    # -----------------------------------------------------------------
    # Step 3: Compute alpha_t = exp(dt_t * A_t)
    # dt: (B, L, H),  A: (H,)  -> alpha: (B, L, H)
    # -----------------------------------------------------------------
    alpha = torch.exp(dt * A.unsqueeze(0).unsqueeze(0))  # (B, L, H)

    # -----------------------------------------------------------------
    # Step 4: Scan loop — plain decay scan over pre-rotated B/C.
    # Layout convention: h is (batch, nheads, headdim, d_state).
    # x has shape (batch, seqlen, nheads, headdim); at step t, x_t is
    # (batch, nheads, headdim) — i.e. x[:, t, :, :].
    # The B/C dot is over d_state; headdim is a parallel (independent) dim.
    # The state is NOT rotated here — the cumulative rotation is already
    # folded into B_rot/C_rot (official C2/C3 convention).
    # -----------------------------------------------------------------
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

        # h_t = alpha_t * h_{t-1} + dt_t * B_rot_t * x_t
        # shapes: (B,H,1,1) * (B,H,P,N) + (B,H,1,1) * (B,H,1,N) * (B,H,P,1)
        h = alpha_t * h + dt_t * B_rot_t * x_t  # (B, H, P, N)

        # --- readout: y_t = C_rot_t^T @ h_t ---
        # C_rot_t: (batch, nheads, d_state) -> (batch, nheads, 1, d_state)
        C_rot_t = C_rot[:, t, :, :].unsqueeze(2)  # (B, H, 1, N)
        # (h * C_rot_t).sum(-1): (B, H, P, N) -> (B, H, P)
        y_t = (h * C_rot_t).sum(-1)  # (B, H, P)
        # store: y is (batch, seqlen, nheads, headdim)
        y[:, t, :, :] = y_t  # (B, H, P) -> y[:,t,:,:]

    return y
