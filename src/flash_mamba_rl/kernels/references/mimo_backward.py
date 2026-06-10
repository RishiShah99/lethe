"""Mamba-3 MIMO selective scan: forward reference and autograd-based backward.

Implements the MIMO SSM from:
  "Mamba-3" (Tri Dao + Albert Gu, ICLR 2026, arXiv:2603.15569)
  Section 3.3, Equations 12-14 and surrounding prose.

SIGNATURE CHANGES FROM OLD STUB
---------------------------------
The old stub accepted (u, delta, A, B, C, D, mix_weight, dy, *, n_heads_in,
n_heads_out, chunk_size) and raised NotImplementedError.  That signature
predated the math resolution and had several incorrect abstractions:

  * "mix_weight" (a single [n_heads_out, n_heads_in] matrix) does not exist
    in Mamba-3.  The resolved math has TWO separate learnable parameter tensors:
      - mimo_x  (shape: nheads, R, headdim)  — psi_j, applied to x BEFORE the
        B-weighted state update
      - mimo_o  (shape: nheads, R, headdim)  — phi_i, applied to output AFTER
        the C readout
    These are data-independent per-head weight vectors, not a head-mixing matrix.
  * "D" (skip connection) is handled by the outer Mamba block, not the SSM scan.
    Removed from both forward and backward.
  * "n_heads_in / n_heads_out" were conflated.  In Mamba-3 there is a single
    rank R for both input and output.
  * alpha (exp(dt*A)) is now passed pre-computed rather than split into (delta, A),
    matching the paper's Eq. 12 notation.  dt is kept as a separate argument
    because it multiplies B in the input term.
  * B and C carry an explicit rank dimension (R), matching the in_proj split
    structure documented in C1 mamba3.py.

FORWARD MATH REFERENCE
-----------------------
Following Eqs 12-14 and the placement rules from Q1b:

  # Step 1: expand x to rank dimension BEFORE SSM (mimo_x applied here)
  x_r[j] = x * mimo_x[h, j, :]           shape: (batch, seqlen, nheads, headdim)

  # Step 2: per-rank scan (Eq. 12)
  h_t^(j) = alpha_t * h_{t-1}^(j) + dt_t * B_t^(j) * x_r_t^(j)
  where h^(j) shape: (batch, nheads, headdim, d_state)

  # Step 3: aggregate state (Eq. 13)
  h_t = sum_{j=0}^{R-1} h_t^(j)

  # Step 4: per-output-rank readout (Eq. 14)
  y_raw_t^(i) = (C_t^(i))^T @ h_t        shape: (batch, seqlen, nheads, headdim)

  # Step 5: down-project with mimo_o AFTER readout (code-only, not in main text)
  y_t = sum_{i=0}^{R-1} y_raw_t^(i) * mimo_o[h, i, :]

BACKWARD ALGORITHMIC SPEC (for future Triton kernel)
-----------------------------------------------------
The oracle uses torch.autograd.grad (no hand-coded gradient formulae).
The analytic backward structure, documented here for the Triton implementor
(index legend: b=batch, l=seq, r/i/j=rank, h=head, p=headdim, s=d_state):

  # Stage 0: through the output mix — dy_raw[i] from y = sum_i y_raw^(i)*phi_i:
  #   dy_raw[b,l,i,h,p] = dy[b,l,h,p] * mimo_o[h,i,p]
  # Stage 1: per-token readout gradient (sum over output ranks; no time mixing
  # here — each token reads its own aggregated state):
  #   dh_agg[b,l,h,p,s] = einsum("blrhp,blrhs->blhps", dy_raw, C)
  # Stage 2: reverse-time scan accumulation (the recurrence h_t = a_t h_{t-1} + ...
  # makes the total state gradient flow backward through time):
  #   dh_total[b,l,h,p,s] = dh_agg[b,l,h,p,s] + alpha[b,l+1,h] * dh_total[b,l+1,h,p,s]
  # Stage 3: input-side grads, per input rank j (contract over headdim p):
  #   dB[b,l,j,h,s]   = dt[b,l,h] * einsum("bhps,bhp->bhs", dh_total[b,l], x_r[b,l,j])
  #   dx_r[b,l,j,h,p] = dt[b,l,h] * einsum("bhps,bjhs->bjhp", dh_total[b,l], B[b,l])
  # Stage 4: per-output-rank C gradient (per token, against the aggregated state):
  #   dC[b,l,i,h,s] = einsum("bhp,bhps->bhs", dy_raw[b,l,i], h_agg[b,l])
  # Stage 5: parameter grads, summed over batch/seq:
  #   dmimo_x[h,j,p] = sum_{b,l} dx_r[b,l,j,h,p] * x[b,l,h,p]
  #   dmimo_o[h,i,p] = sum_{b,l} dy[b,l,h,p] * y_raw[b,l,i,h,p]

UNRESOLVED / OUT OF SCOPE
--------------------------
  * B_bias / C_bias (observed in C1 but role not stated in main paper text).
  * RoPE rotation on B and C: belongs to the complex_scan_rope oracle, not here.
    This oracle takes pre-rotated B and C as inputs.
  * D skip connection: outer block, not the scan.
  * Trapezoidal lambda term: oracle implements Eq. 12 base form (alpha only).
"""

from typing import NamedTuple

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


def reference_mimo_forward(
    x: Tensor,
    B: Tensor,
    C: Tensor,
    dt: Tensor,
    alpha: Tensor,
    mimo_x: Tensor,
    mimo_o: Tensor,
) -> Tensor:
    """Mamba-3 MIMO SSM forward pass (Eqs 12-14).

    Shape contracts
    ---------------
    x      : (batch, seqlen, nheads, headdim)          float32
    B      : (batch, seqlen, R, nheads, d_state)       float32  [pre-rotated]
    C      : (batch, seqlen, R, nheads, d_state)       float32  [pre-rotated]
    dt     : (batch, seqlen, nheads)                   float32, positive
    alpha  : (batch, seqlen, nheads)                   float32, in (0, 1)
             alpha_t = exp(dt_t * A_t), passed pre-computed
    mimo_x : (nheads, R, headdim)                      float32  psi_j
    mimo_o : (nheads, R, headdim)                      float32  phi_i

    Returns
    -------
    y : (batch, seqlen, nheads, headdim)  float32

    Notes
    -----
    * Explicit Python scan loop over t (oracle, not kernel).
    * B and C are taken as already RoPE-rotated; rotation is handled by
      the complex_scan_rope oracle separately.
    * No D skip, no B/C bias, no trapezoidal lambda.
    * float64 is accepted alongside float32 so torch.autograd.gradcheck can
      validate this exact function (half precision stays rejected).
    """
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"Expected float32/float64 input, got {x.dtype}")

    batch, seqlen, nheads, headdim = x.shape
    R = B.shape[2]
    d_state = B.shape[4]

    # Step 1: Expand x to rank dimension (mimo_x applied BEFORE B projection).
    # x: (B, L, H, P) -> x_r: (B, L, R, H, P)
    # mimo_x: (H, R, P) -> rearrange to (1, 1, R, H, P) for broadcast
    mimo_x_bc = mimo_x.permute(1, 0, 2).unsqueeze(0).unsqueeze(0)  # (1, 1, R, H, P)
    x_r = x.unsqueeze(2) * mimo_x_bc  # (B, L, R, H, P)

    # Per-rank hidden state: shape (batch, R, nheads, headdim, d_state)
    h = torch.zeros(batch, R, nheads, headdim, d_state, dtype=x.dtype, device=x.device)
    y = torch.empty_like(x)

    for t in range(seqlen):
        # alpha_t: (batch, nheads) -> (batch, 1, nheads, 1, 1)
        alpha_t = alpha[:, t, :].unsqueeze(1).unsqueeze(-1).unsqueeze(-1)

        # dt_t: (batch, nheads) -> (batch, 1, nheads, 1, 1)
        dt_t = dt[:, t, :].unsqueeze(1).unsqueeze(-1).unsqueeze(-1)

        # B_t: (batch, R, nheads, d_state) -> (batch, R, nheads, 1, d_state)
        B_t = B[:, t, :, :, :].unsqueeze(3)  # (B, R, H, 1, N)

        # x_r_t: (batch, R, nheads, headdim) -> (batch, R, nheads, headdim, 1)
        x_r_t = x_r[:, t, :, :, :].unsqueeze(-1)  # (B, R, H, P, 1)

        # Eq. 12: h_t^(j) = alpha_t * h_{t-1}^(j) + dt_t * B_t^(j) * x_r_t^(j)
        # shapes: (B,1,H,1,1)*(B,R,H,P,N) + (B,1,H,1,1)*(B,R,H,1,N)*(B,R,H,P,1)
        h = alpha_t * h + dt_t * B_t * x_r_t  # (B, R, H, P, N)

        # Eq. 13: h_agg = sum_{j} h^(j)
        h_agg = h.sum(dim=1)  # (B, H, P, N)

        # Eq. 14: y_raw^(i) = C^(i)^T @ h_agg
        # C_t: (batch, R, nheads, d_state) -> (batch, R, nheads, 1, d_state)
        C_t = C[:, t, :, :, :].unsqueeze(3)  # (B, R, H, 1, N)
        # h_agg broadcast: (B, H, P, N) -> (B, 1, H, P, N)
        h_agg_bc = h_agg.unsqueeze(1)  # (B, 1, H, P, N)
        # dot over d_state: (B, R, H, P)
        y_raw = (h_agg_bc * C_t).sum(-1)  # (B, R, H, P)

        # Step 5: y_t = sum_i y_raw^(i) * mimo_o[h, i, :]
        # mimo_o: (H, R, P) -> (1, R, H, P)
        mimo_o_bc = mimo_o.permute(1, 0, 2).unsqueeze(0)  # (1, R, H, P)
        # sum over R: (B, H, P)
        y_t = (y_raw * mimo_o_bc).sum(1)  # (B, H, P)
        y[:, t, :, :] = y_t  # (B, H, P)

    return y


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------


class MimoGrads(NamedTuple):
    """Gradient bundle returned by the MIMO backward reference."""

    grad_x: Tensor
    grad_B: Tensor
    grad_C: Tensor
    grad_dt: Tensor
    grad_alpha: Tensor
    grad_mimo_x: Tensor
    grad_mimo_o: Tensor


def reference_mimo_backward(
    x: Tensor,
    B: Tensor,
    C: Tensor,
    dt: Tensor,
    alpha: Tensor,
    mimo_x: Tensor,
    mimo_o: Tensor,
    dy: Tensor,
) -> MimoGrads:
    """Mamba-3 MIMO SSM backward pass delegated entirely to torch.autograd.

    Wraps ``reference_mimo_forward`` with ``requires_grad=True`` leaves,
    calls ``torch.autograd.grad`` with the upstream gradient ``dy``, and
    returns gradients as a MimoGrads named tuple.

    The analytic backward structure (two-stage einsum) is documented in the
    module docstring for use when implementing the Triton kernel.  This oracle
    uses autograd — correctness by construction, not by hand-derived formulae.

    Shape contracts  (same as reference_mimo_forward)
    ---------------
    x      : (batch, seqlen, nheads, headdim)          float32
    B      : (batch, seqlen, R, nheads, d_state)       float32
    C      : (batch, seqlen, R, nheads, d_state)       float32
    dt     : (batch, seqlen, nheads)                   float32
    alpha  : (batch, seqlen, nheads)                   float32
    mimo_x : (nheads, R, headdim)                      float32
    mimo_o : (nheads, R, headdim)                      float32
    dy     : (batch, seqlen, nheads, headdim)          float32  upstream gradient

    Returns
    -------
    MimoGrads named tuple with fields:
    ``grad_x``, ``grad_B``, ``grad_C``, ``grad_dt``, ``grad_alpha``,
    ``grad_mimo_x``, ``grad_mimo_o`` — each matching the shape of the
    corresponding input.
    """
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"Expected float32/float64 input, got {x.dtype}")

    # Detach from any existing graph and create fresh leaf tensors.
    x_l = x.detach().requires_grad_(True)
    B_l = B.detach().requires_grad_(True)
    C_l = C.detach().requires_grad_(True)
    dt_l = dt.detach().requires_grad_(True)
    alpha_l = alpha.detach().requires_grad_(True)
    mimo_x_l = mimo_x.detach().requires_grad_(True)
    mimo_o_l = mimo_o.detach().requires_grad_(True)

    y = reference_mimo_forward(x_l, B_l, C_l, dt_l, alpha_l, mimo_x_l, mimo_o_l)

    grads = torch.autograd.grad(
        outputs=y,
        inputs=(x_l, B_l, C_l, dt_l, alpha_l, mimo_x_l, mimo_o_l),
        grad_outputs=dy,
    )

    return MimoGrads(
        grad_x=grads[0],
        grad_B=grads[1],
        grad_C=grads[2],
        grad_dt=grads[3],
        grad_alpha=grads[4],
        grad_mimo_x=grads[5],
        grad_mimo_o=grads[6],
    )
