"""Mamba-3 MIMO selective scan backward pass (stub).

Mamba-3 (Tri Dao + Albert Gu, ICLR 2026) generalises the SISO scan to a
multi-input/multi-output (MIMO) setting via an R²-parallel decomposition:
the full D_in x D_out state-space is factored into ``n_heads_in x n_heads_out``
independent SISO scans that run in parallel, then their outputs are mixed with
a learned head-combination matrix.  The backward pass through this structure
requires accumulating gradients across both the parallel SISO backward sweeps
and the head-mixing linear layer.

This stub documents the intended signature and math.  The full implementation
requires Section 3.2 ("MIMO decomposition") of the Mamba-3 paper and access to
the parallel-scan primitive's analytical backward — to be filled in once the
paper is available.
"""

from torch import Tensor


def reference_mimo_backward(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor,
    mix_weight: Tensor,
    dy: Tensor,
    *,
    n_heads_in: int,
    n_heads_out: int,
    chunk_size: int = 64,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """MIMO selective scan backward pass — NOT YET IMPLEMENTED.

    In Mamba-3 the MIMO SSM is parameterised as ``n_heads_in x n_heads_out``
    parallel SISO channels.  Each channel has its own (A_i, B_i, C_i) triple,
    and the outputs are recombined via ``mix_weight`` (shape
    [n_heads_out, n_heads_in]) before being projected back to model dimension.

    Expected tensor shapes
    ----------------------
    u           : [B, L, D_in]          — multi-dim input
    delta       : [B, L, D_in]          — input-dependent timescales
    A           : [n_heads_in, N]       — per-head log-neg SSM eigenvalues
    B           : [B, L, n_heads_in, N] — per-head input projections
    C           : [B, L, n_heads_out, N]— per-head output projections
    D           : [D_out]               — skip connection weight
    mix_weight  : [n_heads_out, n_heads_in] — MIMO head-mixing matrix
    dy          : [B, L, D_out]         — upstream gradient

    Returns (grad_u, grad_delta, grad_A, grad_B, grad_C, grad_D, grad_mix).

    Open math questions
    -------------------
    1. The exact form of the R²-decomposition cross-terms between heads is not
       fully specified without the paper; gradient accumulation order across
       heads may matter for numerical accuracy.
    2. It is unclear whether ``mix_weight`` is applied before or after the
       per-head output projections C — this changes the backward graph.
    3. Chunked-parallel scan gradients for multi-head layouts may require a
       dedicated parallel prefix-sum backward, not just the SISO autograd
       wrapper used in ``reference_backward_selective_scan``.

    Args:
        u:           Multi-dim input, shape [B, L, D_in], float32.
        delta:       Timescale input, shape [B, L, D_in], float32.
        A:           Per-head log-magnitude matrix, shape [n_heads_in, N].
        B:           Per-head input proj, shape [B, L, n_heads_in, N].
        C:           Per-head output proj, shape [B, L, n_heads_out, N].
        D:           Skip weight, shape [D_out], float32.
        mix_weight:  Head-combination matrix, shape [n_heads_out, n_heads_in].
        dy:          Upstream gradient, shape [B, L, D_out], float32.
        n_heads_in:  Number of input heads (R in R²).
        n_heads_out: Number of output heads (R in R²).
        chunk_size:  Chunk size for the underlying scan.

    Returns:
        Tuple of gradients in order:
        (grad_u, grad_delta, grad_A, grad_B, grad_C, grad_D, grad_mix_weight).

    Raises:
        NotImplementedError: Always — pending Mamba-3 paper (Section 3.2).
    """
    raise NotImplementedError(
        "reference_mimo_backward is not yet implemented. "
        "Requires Mamba-3 (Dao & Gu, ICLR 2026) Section 3.2 — "
        "R² parallel SISO decomposition and MIMO head-mixing backward."
    )
