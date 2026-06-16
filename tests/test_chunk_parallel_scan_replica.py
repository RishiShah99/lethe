"""Pin the chunk-parallel-carry forward algebra against the serial oracle.

The serial-L forward scan (``reference_forward_chunked_scan``, mirrored by the
shipped C1 Triton kernel) carries the SSM state step-by-step over all L. The
chunk-parallel-carry restructuring is the deferred long-L speedup lever: it
splits L into chunks of size K and reassociates the *same* linear recurrence
into three phases, exposing parallelism across chunks where the serial form has
none. It is a reassociation, not new math (the SSD chunked-scan decomposition,
Dao & Gu 2024 / official ``mamba`` ``chunk_scan``):

    h_t = a_t * h_{t-1} + bu_t          (bu_t = b_bar_t * u_t, h_{-1}=0)

  Phase 1 (local, parallel across chunks): per chunk c, run the K-step scan
    from a zero state -> hloc[c,j]; the chunk's end state Sloc[c]=hloc[c,K-1]
    and its total decay A_chunk[c]=prod_j a[c,j] (inclusive cumprod P_incl).
  Phase 2 (carry, serial over the nc=L/K chunks, not over L):
    hin[0]=0; hin[c] = A_chunk[c-1]*hin[c-1] + Sloc[c-1].
  Phase 3 (combine): h[c,j] = P_incl[c,j]*hin[c] + hloc[c,j].

Phase 3 is exact algebraically; in fp it differs from the serial thread only by
the distributive split a*(x+y) -> a*x+a*y, i.e. within the eps*sqrt(chain)*scale
reduction-order band the gates already allow. This test holds the replica to
that band so the Triton kernel that mirrors it has a pinned, hardware-free
correctness target. The within-chunk and cross-chunk loops are serial in K and
nc respectively but vectorised over (batch, chunk, D, N) — the shape the GPU
kernel parallelises.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from flash_mamba_rl.kernels.ops.forward_chunked_scan import _scan_eager
from flash_mamba_rl.kernels.references.forward_chunked_scan import (
    reference_forward_chunked_scan,
)


def chunk_parallel_scan_replica(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor,
    *,
    chunk_size: int,
) -> Tensor:
    """Chunk-parallel-carry forward scan, op-for-op as the Triton kernel will."""
    batch, seq_len, d_model = u.shape
    n_state = A.shape[1]
    if seq_len % chunk_size != 0:
        raise ValueError(f"seq_len {seq_len} must be divisible by chunk_size {chunk_size}")
    nc = seq_len // chunk_size
    k = chunk_size

    delta_bar = F.softplus(delta)
    a_bar = torch.exp(delta_bar.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # [B,L,D,N]
    bu = (delta_bar.unsqueeze(-1) * B.unsqueeze(2)) * u.unsqueeze(-1)  # [B,L,D,N]

    a_c = a_bar.reshape(batch, nc, k, d_model, n_state)
    bu_c = bu.reshape(batch, nc, k, d_model, n_state)
    c_c = C.reshape(batch, nc, k, n_state)
    u_c = u.reshape(batch, nc, k, d_model)

    # Phase 1 — local scan from zero state, serial in j, parallel across chunks.
    hloc = torch.zeros(batch, nc, d_model, n_state, dtype=u.dtype, device=u.device)
    hloc_all = torch.empty(batch, nc, k, d_model, n_state, dtype=u.dtype, device=u.device)
    for j in range(k):
        hloc = a_c[:, :, j] * hloc + bu_c[:, :, j]
        hloc_all[:, :, j] = hloc
    p_incl = torch.cumprod(a_c, dim=2)  # [B,nc,K,D,N]
    a_chunk = p_incl[:, :, k - 1]  # [B,nc,D,N]
    s_loc = hloc_all[:, :, k - 1]  # [B,nc,D,N]

    # Phase 2 — cross-chunk carry, serial over the nc chunk boundaries.
    hin_all = torch.empty(batch, nc, d_model, n_state, dtype=u.dtype, device=u.device)
    carry = torch.zeros(batch, d_model, n_state, dtype=u.dtype, device=u.device)
    for c in range(nc):
        hin_all[:, c] = carry
        carry = a_chunk[:, c] * carry + s_loc[:, c]

    # Phase 3 — combine the carry-in contribution with the local scan, then read out.
    h_full = p_incl * hin_all.unsqueeze(2) + hloc_all  # [B,nc,K,D,N]
    y = (h_full * c_c.unsqueeze(3)).sum(-1) + D * u_c  # [B,nc,K,D]
    return y.reshape(batch, seq_len, d_model)


def _draw(
    batch: int, seq_len: int, d_model: int, n_state: int, seed: int, dtype: torch.dtype
) -> tuple[Tensor, ...]:
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(batch, seq_len, d_model, generator=g, dtype=dtype)
    delta = torch.randn(batch, seq_len, d_model, generator=g, dtype=dtype) - 4.0  # near-integrator
    A = -torch.rand(d_model, n_state, generator=g, dtype=dtype) - 0.5  # log-neg
    B = torch.randn(batch, seq_len, n_state, generator=g, dtype=dtype)
    C = torch.randn(batch, seq_len, n_state, generator=g, dtype=dtype)
    D = torch.randn(d_model, generator=g, dtype=dtype)
    return u, delta, A, B, C, D


@pytest.mark.parametrize(
    ("batch", "seq_len", "d_model", "n_state", "chunk_size"),
    [
        (2, 256, 8, 16, 64),
        (1, 2048, 4, 8, 128),
        (2, 4096, 6, 16, 256),
        (1, 16384, 4, 8, 256),  # long-L: the regime the restructuring targets
        (3, 384, 5, 12, 64),  # nc not a power of two, d/n not aligned
    ],
)
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_chunk_parallel_matches_reference_fp32(
    batch: int, seq_len: int, d_model: int, n_state: int, chunk_size: int, seed: int
) -> None:
    u, delta, A, B, C, D = _draw(batch, seq_len, d_model, n_state, seed, torch.float32)
    want = reference_forward_chunked_scan(u, delta, A, B, C, D, chunk_size=chunk_size)
    got = chunk_parallel_scan_replica(u, delta, A, B, C, D, chunk_size=chunk_size)
    # Reassociation -> eps*sqrt(chain)*scale band, not bitwise.
    scale = want.abs().max().clamp_min(1.0)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=2e-5 * scale)


@pytest.mark.parametrize("chunk_size", [16, 64, 256])
def test_chunk_size_invariance_fp64(chunk_size: int) -> None:
    # fp64: the reassociation error collapses, so the result is chunk-size-free
    # to a far tighter band — the algebra itself is exact. The reference is
    # fp32-only, so oracle against the fp64-capable eager path (same math).
    u, delta, A, B, C, D = _draw(2, 1024, 6, 16, seed=3, dtype=torch.float64)
    want = _scan_eager(u, delta, A, B, C, D)
    got = chunk_parallel_scan_replica(u, delta, A, B, C, D, chunk_size=chunk_size)
    torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-10)


def test_indivisible_chunk_rejected() -> None:
    u, delta, A, B, C, D = _draw(1, 100, 4, 8, seed=0, dtype=torch.float32)
    with pytest.raises(ValueError, match="divisible"):
        chunk_parallel_scan_replica(u, delta, A, B, C, D, chunk_size=64)
