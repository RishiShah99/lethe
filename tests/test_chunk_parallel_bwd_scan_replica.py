"""Pin the chunk-parallel-carry SISO backward algebra against the serial oracle.

The serial-L backward (``reference_backward_selective_scan`` = autograd through
the forward; mirrored op-for-op by the shipped C2 Triton kernel
``_triton_bwd_scan``) carries two states step-by-step over all L: the forward
state ``h`` (for the per-step ``h_{t-1}``/``h_t`` the gradient formulae need) and
the reverse adjoint ``g_t = dL/dh_t``. Both are linear recurrences, so both
reassociate the same way the forward scan does — this is the deferred long-L
speedup lever for the *backward*, the half that makes the whole training path
long-L competitive.

The forward state is the C1 recurrence (``test_chunk_parallel_scan_replica``).
The reverse adjoint is its mirror image — a linear scan running newest-first:

    g_t = dy_t * C_t + a_{t+1} * g_{t+1}          (a = a_bar, g_L = 0)

i.e. the same form as ``h_t = a_t*h_{t-1} + bu_t`` read in reverse time. So the
SSD chunked-carry decomposition applies verbatim (Dao & Gu 2024 / official
``mamba``; not new math):

  Phase 1 (local, parallel across chunks): per chunk c, run the K-step reverse
    scan from a zero carry -> gloc[c,j]; the chunk's carry-out at its oldest edge
    Sg[c] = a_{c,0}*gloc[c,0] and its total decay A_chunk[c] = prod_j a_{c,j}.
  Phase 2 (carry, serial over the nc=L/K chunks, newest-first):
    Gin[nc-1]=0; Gin[c] = A_chunk[c+1]*Gin[c+1] + Sg[c+1].
  Phase 3 (combine): g[c,j] = Pdec[c,j]*Gin[c] + gloc[c,j], where the carry-in
    decay Pdec[c,j] = prod_{i>j} a_{c,i} is the exclusive reverse cumprod of a.

Once ``h`` (forward chunk-parallel) and ``g`` (reverse chunk-parallel) are in
hand, every gradient is a local elementwise/reduction expression — computed
here exactly grouping-by-grouping as the serial kernel does (``gm=(g*h_{t-1})*a``
once; the two delta paths as separate N-reductions; grad_A=sum gm*dbar; etc.).
Phase 3 is algebraically exact; in fp it differs from the serial thread only by
the distributive split, i.e. within the eps*sqrt(chain)*scale reduction-order
band the gates already allow. This test holds the replica to that band so the
Triton kernel that mirrors it has a pinned, hardware-free correctness target.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TypeVar

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from lethe.kernels.ops.forward_chunked_scan import _scan_eager
from lethe.kernels.references.backward_selective_scan import (
    SelectiveScanGrads,
    reference_backward_selective_scan,
)

_T = TypeVar("_T")


def _in_big_stack(fn: Callable[[], _T]) -> _T:
    """Run ``fn`` in a worker thread with a 64 MiB C stack.

    The reference backward is autograd through a depth-L Python-loop forward;
    at L=16384 the engine's graph traversal overflows the default thread stack
    (Windows especially). This is a test-harness detail — the chunk-parallel
    algebra itself is depth-free.
    """
    box: dict[str, _T] = {}

    def work() -> None:
        box["v"] = fn()

    old = threading.stack_size(64 * 1024 * 1024)
    try:
        t = threading.Thread(target=work)
        t.start()
        t.join()
    finally:
        threading.stack_size(old)
    return box["v"]


def chunk_parallel_bwd_scan_replica(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor,
    dy: Tensor,
    *,
    chunk_size: int,
) -> SelectiveScanGrads:
    """Chunk-parallel-carry SISO backward, op-for-op as the Triton kernel will.

    fp64 inputs stay fp64 (exact-algebra check); everything else computes in
    fp32 and rounds once per gradient output, the kernel's mixed-precision
    contract.
    """
    batch, seq_len, d_model = u.shape
    n_state = A.shape[1]
    if seq_len % chunk_size != 0:
        raise ValueError(f"seq_len {seq_len} must be divisible by chunk_size {chunk_size}")
    nc = seq_len // chunk_size
    k = chunk_size

    cdt = torch.float64 if u.dtype == torch.float64 else torch.float32
    uc, dc = u.to(cdt), delta.to(cdt)
    ac, bc, cc, dsc, dyc = A.to(cdt), B.to(cdt), C.to(cdt), D.to(cdt), dy.to(cdt)

    delta_bar = F.softplus(dc)  # [B,L,D]
    a_bar = torch.exp(delta_bar.unsqueeze(-1) * ac.unsqueeze(0).unsqueeze(0))  # [B,L,D,N]
    b_bar = delta_bar.unsqueeze(-1) * bc.unsqueeze(2)  # [B,L,D,N]
    bu = b_bar * uc.unsqueeze(-1)  # [B,L,D,N]

    a_c = a_bar.reshape(batch, nc, k, d_model, n_state)
    bu_c = bu.reshape(batch, nc, k, d_model, n_state)
    dyc_c = (dyc.unsqueeze(-1) * cc.unsqueeze(2)).reshape(batch, nc, k, d_model, n_state)

    # ---- Forward h-states via chunk-parallel carry (h_t for every t). ----
    hloc = torch.zeros(batch, nc, d_model, n_state, dtype=cdt, device=u.device)
    hloc_all = torch.empty(batch, nc, k, d_model, n_state, dtype=cdt, device=u.device)
    for j in range(k):
        hloc = a_c[:, :, j] * hloc + bu_c[:, :, j]
        hloc_all[:, :, j] = hloc
    p_incl = torch.cumprod(a_c, dim=2)  # [B,nc,K,D,N] inclusive forward cumprod
    a_chunk = p_incl[:, :, k - 1]  # [B,nc,D,N] = prod_j a_{c,j}
    s_loc = hloc_all[:, :, k - 1]

    hin = torch.empty(batch, nc, d_model, n_state, dtype=cdt, device=u.device)
    carry = torch.zeros(batch, d_model, n_state, dtype=cdt, device=u.device)
    for c in range(nc):
        hin[:, c] = carry
        carry = a_chunk[:, c] * carry + s_loc[:, c]
    h_full = (p_incl * hin.unsqueeze(2) + hloc_all).reshape(batch, seq_len, d_model, n_state)
    # h_{t-1}: the state entering step t (h_{-1}=0).
    h_prev = torch.cat(
        [torch.zeros(batch, 1, d_model, n_state, dtype=cdt, device=u.device), h_full[:, :-1]],
        dim=1,
    )

    # ---- Reverse g-states via chunk-parallel carry (g_t for every t). ----
    gloc = torch.zeros(batch, nc, d_model, n_state, dtype=cdt, device=u.device)
    gloc_all = torch.empty(batch, nc, k, d_model, n_state, dtype=cdt, device=u.device)
    for jj in range(k):
        j = k - 1 - jj
        ag = a_c[:, :, j + 1] * gloc if j < k - 1 else 0.0
        gloc = dyc_c[:, :, j] + ag
        gloc_all[:, :, j] = gloc
    sg = a_c[:, :, 0] * gloc_all[:, :, 0]  # [B,nc,D,N] chunk reverse carry-out

    # Pdec[c,j] = prod_{i>j} a_{c,i}, the exclusive reverse cumprod of a within a chunk.
    rev_incl = torch.cumprod(a_c.flip(2), dim=2)
    ones = torch.ones(batch, nc, 1, d_model, n_state, dtype=cdt, device=u.device)
    pdec = torch.cat([ones, rev_incl[:, :, :-1]], dim=2).flip(2)

    gin = torch.empty(batch, nc, d_model, n_state, dtype=cdt, device=u.device)
    gcarry = torch.zeros(batch, d_model, n_state, dtype=cdt, device=u.device)
    for ci in range(nc):
        c = nc - 1 - ci
        gin[:, c] = gcarry
        gcarry = a_chunk[:, c] * gcarry + sg[:, c]
    g_full = (pdec * gin.unsqueeze(2) + gloc_all).reshape(batch, seq_len, d_model, n_state)

    # ---- Local gradients (grouping-by-grouping as the serial kernel). ----
    dbar = delta_bar  # [B,L,D]
    grad_C = (dyc.unsqueeze(-1) * h_full).sum(dim=2)  # [B,L,N]
    grad_B = (g_full * (uc * dbar).unsqueeze(-1)).sum(dim=2)  # [B,L,N]
    grad_u = (g_full * b_bar).sum(dim=-1) + dsc * dyc  # [B,L,D]

    gm = (g_full * h_prev) * a_bar  # [B,L,D,N]
    ddbar = (gm * ac.unsqueeze(0).unsqueeze(0)).sum(dim=-1) + (
        (g_full * uc.unsqueeze(-1)) * bc.unsqueeze(2)
    ).sum(dim=-1)
    z = torch.exp(dc)
    dsig = torch.where(dc > 20.0, torch.ones_like(dc), z / (z + 1.0))
    grad_delta = ddbar * dsig  # [B,L,D]

    grad_A = (gm * dbar.unsqueeze(-1)).sum(dim=(0, 1))  # [D,N]
    grad_D = (dyc * uc).sum(dim=(0, 1))  # [D]

    return SelectiveScanGrads(
        grad_u=grad_u.to(u.dtype),
        grad_delta=grad_delta.to(u.dtype),
        grad_A=grad_A.to(u.dtype),
        grad_B=grad_B.to(u.dtype),
        grad_C=grad_C.to(u.dtype),
        grad_D=grad_D.to(u.dtype),
    )


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
    dy = torch.randn(batch, seq_len, d_model, generator=g, dtype=dtype)
    return u, delta, A, B, C, D, dy


_FIELDS = ("grad_u", "grad_delta", "grad_A", "grad_B", "grad_C", "grad_D")


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
def test_chunk_parallel_bwd_matches_reference_fp32(
    batch: int, seq_len: int, d_model: int, n_state: int, chunk_size: int, seed: int
) -> None:
    u, delta, A, B, C, D, dy = _draw(batch, seq_len, d_model, n_state, seed, torch.float32)
    want = _in_big_stack(
        lambda: reference_backward_selective_scan(u, delta, A, B, C, D, dy, chunk_size=chunk_size)
    )
    got = chunk_parallel_bwd_scan_replica(u, delta, A, B, C, D, dy, chunk_size=chunk_size)
    for field in _FIELDS:
        w = getattr(want, field)
        scale = w.abs().max().clamp_min(1.0)
        torch.testing.assert_close(
            getattr(got, field), w, rtol=1e-4, atol=3e-5 * scale, msg=f"{field} mismatch"
        )


@pytest.mark.parametrize("chunk_size", [16, 64, 256])
def test_chunk_size_invariance_fp64(chunk_size: int) -> None:
    # fp64: the reassociation error collapses, so the gradients are chunk-size
    # free to a far tighter band — the algebra itself is exact. The fp32-only
    # reference can't oracle fp64, so compare chunk-parallel against the serial
    # autograd through the same (fp64-capable) eager forward.
    u, delta, A, B, C, D, dy = _draw(2, 1024, 6, 16, seed=3, dtype=torch.float64)
    leaves = [t.detach().requires_grad_(True) for t in (u, delta, A, B, C, D)]
    y = _scan_eager(*leaves)
    want = torch.autograd.grad(y, leaves, dy)
    got = chunk_parallel_bwd_scan_replica(u, delta, A, B, C, D, dy, chunk_size=chunk_size)
    for field, w in zip(_FIELDS, want, strict=True):
        torch.testing.assert_close(getattr(got, field), w, rtol=1e-9, atol=1e-9, msg=field)


def test_indivisible_chunk_rejected() -> None:
    u, delta, A, B, C, D, dy = _draw(1, 100, 4, 8, seed=0, dtype=torch.float32)
    with pytest.raises(ValueError, match="divisible"):
        chunk_parallel_bwd_scan_replica(u, delta, A, B, C, D, dy, chunk_size=64)


class TestChunkParallelBwdNonFinites:
    """EXC-01 pin: the chunked reverse carry must mint the same NaN/Inf pattern
    autograd's serial adjoint does. The combine form (Pdec*Gin) exercises the
    cumulative-product machinery most prone to 0*Inf, so a match here is the
    strict evidence the gate battery needs before the B200."""

    def _masks_match(self, inject: str) -> None:
        u, delta, A, B, C, D, dy = _draw(1, 64, 6, 8, seed=3, dtype=torch.float32)
        if inject == "inf":
            dy[0, 32, 2] = float("inf")
        elif inject == "ninf":
            dy[0, 20, 1] = float("-inf")
        elif inject == "nan":
            dy[0, 32, 4] = float("nan")
        want = _in_big_stack(
            lambda: reference_backward_selective_scan(u, delta, A, B, C, D, dy, chunk_size=16)
        )
        got = chunk_parallel_bwd_scan_replica(u, delta, A, B, C, D, dy, chunk_size=16)
        for field in _FIELDS:
            w, g = getattr(want, field), getattr(got, field)
            assert torch.equal(w.isnan(), g.isnan()), f"{field}: NaN mask diverges ({inject})"
            assert torch.equal(w.isinf(), g.isinf()), f"{field}: Inf mask diverges ({inject})"

    @pytest.mark.parametrize("inject", ["inf", "ninf", "nan"])
    def test_non_finite_dy_mask_parity(self, inject: str) -> None:
        self._masks_match(inject)


def test_single_chunk_equals_serial_sweep() -> None:
    # nc=1 (chunk_size==L) reduces the carry to a no-op: Gin=0, the local reverse
    # sweep IS the full serial adjoint. A direct check that Phase 1 alone is the
    # serial backward when there is only one chunk.
    u, delta, A, B, C, D, dy = _draw(2, 64, 4, 8, seed=5, dtype=torch.float64)
    leaves = [t.detach().requires_grad_(True) for t in (u, delta, A, B, C, D)]
    y = _scan_eager(*leaves)
    want = torch.autograd.grad(y, leaves, dy)
    got = chunk_parallel_bwd_scan_replica(u, delta, A, B, C, D, dy, chunk_size=64)
    for field, w in zip(_FIELDS, want, strict=True):
        torch.testing.assert_close(getattr(got, field), w, rtol=1e-9, atol=1e-9, msg=field)
