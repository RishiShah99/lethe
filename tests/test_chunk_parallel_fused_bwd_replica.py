"""Pin the chunk-parallel-carry C6 fused-block backward algebra.

The serial C6 backward (``_triton_fused_block_bwd``; mirrored by
``test_fused_block_bwd_kernel_replica``) walks two serial-L sweeps: a forward
re-stage that register-carries the scan state ``h`` across chunks, and a reverse
sweep that register-carries the adjoint ``ag = a_bar*g``. Both are the same
linear recurrences C1/C2 carry, so both reassociate the same way — this extends
the long-L lever from the SISO backward (C2) to the fused training block.

  - forward ``h_t = a_t*h_{t-1} + (dbar*z)_t*B_t`` (z = conv->SiLU): local
    per-chunk reduce -> ``Sloc``/``Adecay`` -> O(L/K) carry -> ``hin`` entering
    each chunk; a per-chunk readout recomputes h and stages ``ys``.
  - reverse ``g_t = dys_t*C_t + a_{t+1}*g_{t+1}`` (dys = RMSNorm-bwd output):
    the C2 reverse twin — local reverse reduce -> ``Sg`` -> O(L/K) newest-first
    carry (the *same* ``Adecay`` product) -> ``Gin`` -> per-chunk reverse readout
    seeded from ``Gin``.

The norm backward (kernel 2) and the gather-form grad_x (kernel 4) are already
chunk-free and carry over unchanged. Every gradient grouping is the serial
replica's, byte-for-byte; only the two carries move from registers to the
chunked decomposition. This holds the replica to the eps*sqrt(chain)*scale band
(fp64 exact) so the Triton kernel that mirrors it has a hardware-free target.
"""

from __future__ import annotations

import math

import pytest
import torch

from lethe.kernels.ops.fused_block_backward import _fused_bwd_eager
from lethe.kernels.references.fused_block_backward import FusedBlockGrads

from .test_fused_block_bwd_kernel_replica import _norm_bwd_replica

_SOFTPLUS_THRESHOLD = 20.0


def chunk_parallel_fused_bwd_replica(
    x: torch.Tensor,
    conv_w: torch.Tensor,
    conv_b: torch.Tensor,
    delta: torch.Tensor,
    a_mat: torch.Tensor,
    b_proj: torch.Tensor,
    c_proj: torch.Tensor,
    d_skip: torch.Tensor,
    norm_w: torch.Tensor,
    dy: torch.Tensor,
    eps: float,
    chunk_k: int,
    block_d: int,
    block_t: int,
) -> FusedBlockGrads:
    """The chunk-parallel C6 pipeline, statement for statement, in torch."""
    batch, seq_in, d_model = x.shape
    conv_k = conv_w.shape[-1]
    l_out = seq_in - (conv_k - 1)
    n_state = a_mat.shape[1]
    n_chunks = l_out // chunk_k
    dt = x.dtype
    w = conv_w[:, 0, :]

    def conv_silu(t: int) -> tuple[torch.Tensor, torch.Tensor]:
        xw = x[:, t : t + conv_k, :].permute(0, 2, 1)
        conv = (w.unsqueeze(0) * xw).sum(-1) + conv_b
        return conv, conv * (1.0 / (1.0 + torch.exp(-conv)))

    def step_coeffs(t: int) -> tuple[torch.Tensor, torch.Tensor]:
        dlt = delta[:, t]
        dbar = torch.where(dlt > _SOFTPLUS_THRESHOLD, dlt, torch.log1p(torch.exp(dlt)))
        abar = torch.exp(dbar.unsqueeze(-1) * a_mat.unsqueeze(0))
        return dbar, abar

    # ---- Forward chunk-parallel carry: local reduce -> Adecay/Sloc -> hin. ----
    s_loc = torch.zeros(n_chunks, batch, d_model, n_state, dtype=dt)
    adecay = torch.zeros(n_chunks, batch, d_model, n_state, dtype=dt)
    for c in range(n_chunks):
        hloc = torch.zeros(batch, d_model, n_state, dtype=dt)
        dec = torch.ones(batch, d_model, n_state, dtype=dt)
        for j in range(chunk_k):
            t = c * chunk_k + j
            _conv, z = conv_silu(t)
            dbar, abar = step_coeffs(t)
            hloc = abar * hloc + (dbar * z).unsqueeze(-1) * b_proj[:, t].unsqueeze(1)
            dec = dec * abar
        s_loc[c] = hloc
        adecay[c] = dec

    hin = torch.zeros(n_chunks, batch, d_model, n_state, dtype=dt)
    carry = torch.zeros(batch, d_model, n_state, dtype=dt)
    for c in range(n_chunks):
        hin[c] = carry
        carry = adecay[c] * carry + s_loc[c]

    # Per-chunk forward readout: recompute h from hin, stage ys.
    ys = torch.empty(batch, l_out, d_model, dtype=dt)
    for c in range(n_chunks):
        h = hin[c].clone()
        for j in range(chunk_k):
            t = c * chunk_k + j
            _conv, z = conv_silu(t)
            dbar, abar = step_coeffs(t)
            h = abar * h + (dbar * z).unsqueeze(-1) * b_proj[:, t].unsqueeze(1)
            ys[:, t] = (h * c_proj[:, t].unsqueeze(1)).sum(-1) + d_skip * z

    # kernel 2: RMSNorm backward (chunk-free, unchanged).
    dys, grad_nw = _norm_bwd_replica(ys, dy, norm_w, eps, block_t, block_d)

    # ---- Reverse chunk-parallel carry: local reverse reduce -> Sg -> Gin. ----
    sg = torch.zeros(n_chunks, batch, d_model, n_state, dtype=dt)
    for c in range(n_chunks):
        g = torch.zeros(batch, d_model, n_state, dtype=dt)
        abar_prev = torch.zeros(batch, d_model, n_state, dtype=dt)
        for jj in range(chunk_k):
            j = chunk_k - 1 - jj
            t = c * chunk_k + j
            _dbar, abar = step_coeffs(t)
            dyc = dys[:, t].unsqueeze(-1) * c_proj[:, t].unsqueeze(1)
            g = dyc + abar_prev * g
            abar_prev = abar
        sg[c] = abar_prev * g  # a_{c,0} * gloc_{c,0}

    gin = torch.zeros(n_chunks, batch, d_model, n_state, dtype=dt)
    gcarry = torch.zeros(batch, d_model, n_state, dtype=dt)
    for ci in range(n_chunks):
        c = n_chunks - 1 - ci
        gin[c] = gcarry
        gcarry = adecay[c] * gcarry + sg[c]

    # kernel 3: per-chunk reverse readout, seeded from hin (forward) + Gin
    # (reverse). The inner body is the serial replica's, unchanged.
    n_d_blocks = (d_model + block_d - 1) // block_d
    dconv = torch.empty(batch, l_out, d_model, dtype=dt)
    grad_delta = torch.empty(batch, l_out, d_model, dtype=dt)
    gb_part = torch.zeros(batch, n_d_blocks, l_out, n_state, dtype=dt)
    gc_part = torch.zeros(batch, n_d_blocks, l_out, n_state, dtype=dt)
    ga_acc = torch.zeros(batch, d_model, n_state, dtype=dt)
    gd_acc = torch.zeros(batch, d_model, dtype=dt)
    gw_acc = torch.zeros(batch, d_model, conv_k, dtype=dt)
    gcb_acc = torch.zeros(batch, d_model, dtype=dt)
    hbuf = torch.empty(chunk_k, batch, d_model, n_state, dtype=dt)
    zbuf = torch.empty(chunk_k, batch, d_model, dtype=dt)
    cvbuf = torch.empty(chunk_k, batch, d_model, dtype=dt)
    d_splits = [slice(d0, d0 + block_d) for d0 in range(0, d_model, block_d)]

    for c in range(n_chunks):
        t0 = c * chunk_k
        h_prev = hin[c].clone()
        for j in range(chunk_k):
            t = t0 + j
            hbuf[j] = h_prev
            conv, z = conv_silu(t)
            cvbuf[j] = conv
            zbuf[j] = z
            dbar, abar = step_coeffs(t)
            h_prev = abar * h_prev + (dbar.unsqueeze(-1) * b_proj[:, t].unsqueeze(1)) * zbuf[
                j
            ].unsqueeze(-1)
        h_cur = h_prev
        ag_carry = gin[c].clone()
        for jj in range(chunk_k):
            j = chunk_k - 1 - jj
            t = t0 + j
            dys_t = dys[:, t]
            dbar, abar = step_coeffs(t)
            bb = dbar.unsqueeze(-1) * b_proj[:, t].unsqueeze(1)
            h_tm1 = hbuf[j]
            z = zbuf[j]
            conv = cvbuf[j]

            g = dys_t.unsqueeze(-1) * c_proj[:, t].unsqueeze(1) + ag_carry
            for i, ds in enumerate(d_splits):
                gc_part[:, i, t] = (dys_t[:, ds].unsqueeze(-1) * h_cur[:, ds]).sum(1)
                gb_part[:, i, t] = (
                    (g[:, ds] * z[:, ds].unsqueeze(-1)) * dbar[:, ds].unsqueeze(-1)
                ).sum(1)

            gz_t = (g * bb).sum(-1) + d_skip * dys_t
            sig = 1.0 / (1.0 + torch.exp(-conv))
            dconv_t = (gz_t * sig) * (1.0 + conv * (1.0 - sig))
            dconv[:, t] = dconv_t

            xw = x[:, t : t + conv_k, :].permute(0, 2, 1)
            gw_acc = gw_acc + dconv_t.unsqueeze(-1) * xw
            gcb_acc = gcb_acc + dconv_t

            gm = (g * h_tm1) * abar
            ddbar = (gm * a_mat.unsqueeze(0)).sum(-1) + (
                (g * z.unsqueeze(-1)) * b_proj[:, t].unsqueeze(1)
            ).sum(-1)
            dlt = delta[:, t]
            zexp = torch.exp(dlt)
            dsig = torch.where(dlt > _SOFTPLUS_THRESHOLD, torch.ones_like(dlt), zexp / (zexp + 1.0))
            grad_delta[:, t] = ddbar * dsig

            ga_acc = ga_acc + gm * dbar.unsqueeze(-1)
            gd_acc = gd_acc + dys_t * z
            ag_carry = abar * g
            h_cur = h_tm1

    # kernel 4: gather-form grad_x.
    grad_x = torch.zeros(batch, seq_in, d_model, dtype=dt)
    for k in range(conv_k):
        grad_x[:, k : k + l_out] += dconv * w[:, k]

    return FusedBlockGrads(
        grad_x=grad_x,
        grad_conv_weight=gw_acc.sum(dim=0).unsqueeze(1),
        grad_conv_bias=gcb_acc.sum(dim=0),
        grad_delta=grad_delta,
        grad_A=ga_acc.sum(dim=0),
        grad_B=gb_part.sum(dim=1),
        grad_C=gc_part.sum(dim=1),
        grad_D=gd_acc.sum(dim=0),
        grad_norm_weight=grad_nw.clone(),
    )


def _inputs(
    b: int, l_out: int, d: int, n: int, k: int, seed: int, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)

    def mk(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, dtype=dtype)

    x = mk(b, l_out + k - 1, d)
    conv_w = mk(d, 1, k) / math.sqrt(k)
    conv_b = 0.5 * mk(d)
    delta = mk(b, l_out, d)
    a_mat = -torch.rand(d, n, dtype=dtype)
    b_proj = mk(b, l_out, n)
    c_proj = mk(b, l_out, n)
    d_skip = mk(d)
    norm_w = 1.0 + 0.25 * mk(d)
    dy = mk(b, l_out, d)
    return x, conv_w, conv_b, delta, a_mat, b_proj, c_proj, d_skip, norm_w, dy


class TestChunkParallelFusedBwdParity:
    def _check(
        self,
        b: int,
        l_out: int,
        d: int,
        n: int,
        k: int,
        seed: int = 0,
        chunk_k: int = 8,
        block_d: int = 4,
        block_t: int = 8,
        saturate: bool = False,
        bound: float = 3e-5,
    ) -> None:
        args = _inputs(b, l_out, d, n, k, seed)
        if saturate:
            args[3].view(-1)[::7] = 25.0
            args[3].view(-1)[::11] = 95.0
        got = chunk_parallel_fused_bwd_replica(*args, 1e-5, chunk_k, block_d, block_t)
        want = _fused_bwd_eager(*args, eps=1e-5)
        for field, g, r in zip(FusedBlockGrads._fields, got, want, strict=True):
            max_err = (g - r).abs().max().item()
            scale = r.abs().max().clamp(min=1.0).item()
            assert max_err / scale < bound, f"{field} diverges: scale_rel={max_err / scale:.3e}"

    def test_standard_window(self) -> None:
        self._check(2, 64, 12, 16, 4, chunk_k=16)

    def test_window_two(self) -> None:
        self._check(2, 48, 8, 16, 2, chunk_k=16)

    def test_window_one_degenerate(self) -> None:
        self._check(1, 32, 6, 8, 1, chunk_k=16)

    def test_many_chunks(self) -> None:
        self._check(2, 256, 8, 16, 4, seed=7, chunk_k=64, bound=1e-4)

    def test_nc_not_power_of_two(self) -> None:
        # l_out=192, chunk_k=64 -> 3 chunks; d/n unaligned to block_d.
        self._check(2, 192, 10, 12, 4, seed=2, chunk_k=64, block_d=4, bound=1e-4)

    def test_saturated_softplus_branch(self) -> None:
        self._check(2, 64, 8, 8, 4, chunk_k=16, saturate=True)

    def test_d_block_tail(self) -> None:
        self._check(2, 64, 10, 8, 4, chunk_k=16, block_d=4)

    def test_fp64_tight(self) -> None:
        args = _inputs(2, 128, 12, 16, 4, seed=11, dtype=torch.float64)
        got = chunk_parallel_fused_bwd_replica(*args, 1e-5, 32, 4, 8)
        want = _fused_bwd_eager(*args, eps=1e-5)
        for field, g, r in zip(FusedBlockGrads._fields, got, want, strict=True):
            max_err = (g - r).abs().max().item()
            assert max_err < 1e-11, f"fp64 {field} diverges: {max_err:.3e}"


class TestChunkParallelFusedBwdNonFinites:
    """The reassociation must mint the same NaN/Inf pattern autograd does."""

    def _masks_match(self, args: tuple[torch.Tensor, ...]) -> None:
        got = chunk_parallel_fused_bwd_replica(*args, 1e-5, 8, 4, 8)
        want = _fused_bwd_eager(*args, eps=1e-5)
        for field, g, r in zip(FusedBlockGrads._fields, got, want, strict=True):
            assert torch.equal(g.isnan(), r.isnan()), f"{field}: NaN mask diverges"
            assert torch.equal(g.isinf(), r.isinf()), f"{field}: Inf mask diverges"

    def test_nan_in_dy(self) -> None:
        args = list(_inputs(1, 16, 6, 8, 4, seed=3))
        args[9][0, 7, 2] = float("nan")
        self._masks_match(tuple(args))

    @pytest.mark.parametrize("val", [float("inf"), float("-inf")])
    def test_inf_in_dy(self, val: float) -> None:
        args = list(_inputs(1, 16, 6, 8, 4, seed=3))
        args[9][0, 5, 1] = val
        self._masks_match(tuple(args))
