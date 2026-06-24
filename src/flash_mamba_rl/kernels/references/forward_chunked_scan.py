"""Mamba-1/3 SISO selective scan forward pass, computed chunk-by-chunk.

Reference oracle for the verifier. Implements the core SSM recurrence from:
  "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
  Gu & Dao, ICLR 2024 (arXiv:2312.00752)

Used as the scan primitive inside Mamba-3 (Lahoti, Li, et al., ICLR 2026).
Correctness over speed: float32 throughout, plain Python loop per chunk.

DISCRETISATION SCOPE (Mamba-3 trapezoidal vs ZOH)
-------------------------------------------------
``reference_forward_chunked_scan`` is the zero-order-hold (ZOH / Euler) rule —
the Mamba-1/2/SSD base case, which is exactly the lambda=1 limit of the Mamba-3
exponential-trapezoidal discretisation. The six verifier-graded kernels (C1-C6)
all implement this ZOH form, so it is the oracle they are graded against.

``reference_forward_trapezoidal_scan`` implements the full data-dependent
trapezoidal discretisation Mamba-3 introduces (paper Prop. 3.2.2, with
lambda = sigmoid(trap) confirmed from official ``mamba3.py::_preprocess``;
sourcing in ``docs/mamba3_math_resolution.md``). It is NOT wired into kernel
grading: the kernels implement lambda=1, so grading them against the trapezoidal
oracle would require pushing the trap term into all six kernels first (a scoped
kernel extension, not an oracle change). It is provided + pinned here as the
resolved Mamba-3 math at the oracle level; ``test_trapezoidal_scan_reference``
proves it reduces byte-identically to the ZOH oracle at lambda=1.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def reference_forward_chunked_scan(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor,
    *,
    chunk_size: int = 64,
) -> Tensor:
    """Selective state-space scan (SISO, Mamba-1 recurrence), chunked forward.

    Discretises (A, B) with the zero-order-hold (ZOH) rule:
        delta_bar = softplus(delta)            [B, L, D]
        A_bar     = exp(delta_bar * A)         [B, L, D, N]  (A is [D, N], log-neg)
        B_bar     = delta_bar[..., None] * B   [B, L, D, N]  (broadcast)

    Recurrence per step t (state h is [B, D, N]):
        h_t = A_bar_t * h_{t-1} + B_bar_t * u_t[..., None]
        y_t = (h_t * C_t).sum(-1) + D * u_t

    Chunks are processed sequentially; within each chunk the recurrence runs
    step-by-step in Python (no parallelism — this is the oracle, not the kernel).

    Args:
        u:          Input tensor, shape [B, L, D], float32.
        delta:      Timescale input (pre-softplus), shape [B, L, D], float32.
        A:          Log-magnitude SSM matrix, shape [D, N], float32, negative.
        B:          Input projection, shape [B, L, N], float32.
        C:          Output projection, shape [B, L, N], float32.
        D:          Skip connection weight, shape [D], float32.
        chunk_size: Number of time-steps per chunk (must divide L evenly).

    Returns:
        y: Output tensor, same shape as u, [B, L, D], float32.

    Raises:
        ValueError: If L is not divisible by chunk_size.
    """
    if u.dtype != torch.float32:
        raise ValueError(f"Expected float32 input, got {u.dtype}")
    batch, seq_len, d_model = u.shape
    n_state = A.shape[1]

    if seq_len % chunk_size != 0:
        raise ValueError(f"seq_len {seq_len} must be divisible by chunk_size {chunk_size}")

    # softplus discretisation of delta  [B, L, D]
    delta_bar = F.softplus(delta)

    # A_bar: [B, L, D, N] = exp(delta_bar[..., None] * A[None, None, :, :])
    a_bar = torch.exp(delta_bar.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))

    # B_bar: [B, L, D, N] = delta_bar[..., None] * B[:, :, None, :]
    b_bar = delta_bar.unsqueeze(-1) * B.unsqueeze(2)

    # Allocate output and running hidden state
    y = torch.empty_like(u)
    h = torch.zeros(batch, d_model, n_state, dtype=u.dtype, device=u.device)

    n_chunks = seq_len // chunk_size
    for chunk_idx in range(n_chunks):
        t0 = chunk_idx * chunk_size
        t1 = t0 + chunk_size
        for t in range(t0, t1):
            # h: [B, D, N]
            h = a_bar[:, t, :, :] * h + b_bar[:, t, :, :] * u[:, t, :].unsqueeze(-1)
            # y_t = sum_n( h * C_t ) + D * u_t
            # C: [B, L, N] -> C[:, t, :] is [B, N]
            y[:, t, :] = (h * C[:, t, :].unsqueeze(1)).sum(-1) + D * u[:, t, :]

    return y


def reference_forward_trapezoidal_scan(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor,
    trap: Tensor,
    *,
    chunk_size: int = 64,
) -> Tensor:
    """Mamba-3 exponential-trapezoidal SISO scan (paper Prop. 3.2.2).

    Generalises :func:`reference_forward_chunked_scan` (the lambda=1 ZOH base
    case) to the data-dependent trapezoidal discretisation Mamba-3 introduces
    over Mamba-2's ZOH. Resolved entirely from sources (no invented math):
    arXiv:2603.15569 Prop. 3.2.2 for the alpha/beta/gamma split, and official
    ``mamba_ssm/modules/mamba3.py::_preprocess`` (``trap = torch.sigmoid(...)``)
    for the lambda gating. Full sourcing: ``docs/mamba3_math_resolution.md``.

    Per step (Delta_t = softplus(delta_t), A negative log-magnitude):
        alpha_t  = exp(Delta_t * A)                       [= a_bar, per (D, N)]
        lambda_t = sigmoid(trap_t)                        in (0, 1)
        beta_t   = (1 - lambda_t) * Delta_t * alpha_t     [weight on the prev token]
        gamma_t  = lambda_t * Delta_t                     [weight on the current token]

        h_t = alpha_t * h_{t-1}
              + gamma_t * B_t   * u_t            (current-token input)
              + beta_t  * B_{t-1} * u_{t-1}      (previous-token input; absent at t=0)
        y_t = sum_n(h_t * C_t) + D * u_t

    At lambda=1 (beta=0, gamma=Delta_t) this is byte-identical to the ZOH oracle
    (pinned by ``test_trapezoidal_reduces_to_zoh_at_lambda_one``). The beta term
    is the only structural addition: a contribution from the previous token's
    B*u, scaled by the current step's decay alpha_t — exactly the trapezoidal
    (vs rectangular/ZOH) integration of the input over [t-1, t].

    Args:
        u:          Input tensor, shape [B, L, D], float32.
        delta:      Timescale input (pre-softplus), shape [B, L, D], float32.
        A:          Log-magnitude SSM matrix, shape [D, N], float32, negative.
        B:          Input projection, shape [B, L, N], float32.
        C:          Output projection, shape [B, L, N], float32.
        D:          Skip connection weight, shape [D], float32.
        trap:       Trapezoidal logits (pre-sigmoid), shape [B, L, D], float32.
        chunk_size: Number of time-steps per chunk (must divide L evenly).

    Returns:
        y: Output tensor, same shape as u, [B, L, D], float32.

    Raises:
        ValueError: If inputs are not float32 or L is not divisible by chunk_size.
    """
    if u.dtype != torch.float32:
        raise ValueError(f"Expected float32 input, got {u.dtype}")
    batch, seq_len, d_model = u.shape
    n_state = A.shape[1]

    if seq_len % chunk_size != 0:
        raise ValueError(f"seq_len {seq_len} must be divisible by chunk_size {chunk_size}")

    delta_bar = F.softplus(delta)  # Delta_t  [B, L, D]
    a_bar = torch.exp(delta_bar.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # [B, L, D, N]
    # b_bar grouped exactly as the ZOH oracle so the lambda=1 limit is bit-identical:
    b_bar = delta_bar.unsqueeze(-1) * B.unsqueeze(2)  # [B, L, D, N]  (= delta_bar * B)
    bu = B.unsqueeze(2) * u.unsqueeze(-1)  # [B, L, D, N]  raw B_t * u_t (previous-token term)
    lam = torch.sigmoid(trap)  # lambda_t  [B, L, D]
    beta = (1.0 - lam) * delta_bar  # [B, L, D]  (alpha_t folded in per-step below)

    y = torch.empty_like(u)
    h = torch.zeros(batch, d_model, n_state, dtype=u.dtype, device=u.device)
    bu_prev = torch.zeros(batch, d_model, n_state, dtype=u.dtype, device=u.device)

    n_chunks = seq_len // chunk_size
    for chunk_idx in range(n_chunks):
        t0 = chunk_idx * chunk_size
        t1 = t0 + chunk_size
        for t in range(t0, t1):
            # current: lambda_t * (delta_bar_t * B_t) * u_t  -> b_bar * u at lambda=1
            cur = lam[:, t, :].unsqueeze(-1) * b_bar[:, t, :, :] * u[:, t, :].unsqueeze(-1)
            # previous: beta_t * alpha_t * (B_{t-1} * u_{t-1})  -> exactly 0 at lambda=1
            prev = beta[:, t, :].unsqueeze(-1) * a_bar[:, t, :, :] * bu_prev
            h = a_bar[:, t, :, :] * h + cur + prev
            y[:, t, :] = (h * C[:, t, :].unsqueeze(1)).sum(-1) + D * u[:, t, :]
            bu_prev = bu[:, t, :, :]

    return y
