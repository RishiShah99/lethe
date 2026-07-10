"""CPU replica of the C6 Triton backward pipeline's exact algorithm."""

from __future__ import annotations

import math

import pytest
import torch

from lethe.kernels.ops.fused_block_backward import _fused_bwd_eager
from lethe.kernels.references.fused_block_backward import FusedBlockGrads

_SOFTPLUS_THRESHOLD = 20.0


def _norm_bwd_replica(
    ys: torch.Tensor,
    dy: torch.Tensor,
    norm_w: torch.Tensor,
    eps: float,
    block_t: int,
    block_d: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Kernel 2's compute order: chunked ssq/sdw, one 1/r^2 after the loop."""
    batch, l_out, d_model = ys.shape
    dys = torch.empty_like(ys)
    gnw_parts = []
    for t0 in range(0, l_out, block_t):
        ys_b = ys[:, t0 : t0 + block_t]
        dy_b = dy[:, t0 : t0 + block_t]
        ssq = torch.zeros(batch, ys_b.shape[1], dtype=ys.dtype)
        sdw = torch.zeros(batch, ys_b.shape[1], dtype=ys.dtype)
        for d0 in range(0, d_model, block_d):
            ys_c = ys_b[:, :, d0 : d0 + block_d]
            dy_c = dy_b[:, :, d0 : d0 + block_d]
            w_c = norm_w[d0 : d0 + block_d]
            ssq = ssq + (ys_c * ys_c).sum(-1)
            sdw = sdw + ((dy_c * w_c) * ys_c).sum(-1)
        r = torch.sqrt(ssq / d_model + eps)
        dm_over_d = (-sdw / (r * r)) / (2.0 * r) / d_model
        gnw_part = torch.zeros(batch, d_model, dtype=ys.dtype)
        for d0 in range(0, d_model, block_d):
            ys_c = ys_b[:, :, d0 : d0 + block_d]
            dy_c = dy_b[:, :, d0 : d0 + block_d]
            w_c = norm_w[d0 : d0 + block_d]
            dys[:, t0 : t0 + block_t, d0 : d0 + block_d] = (dy_c * w_c) / r.unsqueeze(-1) + (
                2.0 * ys_c
            ) * dm_over_d.unsqueeze(-1)
            gnw_part[:, d0 : d0 + block_d] = (dy_c * (ys_c / r.unsqueeze(-1))).sum(1)
        gnw_parts.append(gnw_part)
    grad_nw = torch.stack(gnw_parts).sum(dim=(0, 1))
    return dys, grad_nw


def _bwd_pipeline_replica(
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
    """The four-kernel pipeline, statement for statement, in torch."""
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

    # kernel 1: forward re-stage, ys plus per-chunk state checkpoints.
    ys = torch.empty(batch, l_out, d_model, dtype=dt)
    ckpts = torch.empty(n_chunks, batch, d_model, n_state, dtype=dt)
    h = torch.zeros(batch, d_model, n_state, dtype=dt)
    for c in range(n_chunks):
        ckpts[c] = h
        for j in range(chunk_k):
            t = c * chunk_k + j
            _conv, z = conv_silu(t)
            dbar, abar = step_coeffs(t)
            h = abar * h + (dbar * z).unsqueeze(-1) * b_proj[:, t].unsqueeze(1)
            ys[:, t] = (h * c_proj[:, t].unsqueeze(1)).sum(-1) + d_skip * z

    # kernel 2: RMSNorm backward.
    dys, grad_nw = _norm_bwd_replica(ys, dy, norm_w, eps, block_t, block_d)

    # kernel 3: newest-first reverse sweep with in-chunk recompute.
    n_d_blocks = (d_model + block_d - 1) // block_d
    dconv = torch.empty(batch, l_out, d_model, dtype=dt)
    grad_delta = torch.empty(batch, l_out, d_model, dtype=dt)
    gb_part = torch.zeros(batch, n_d_blocks, l_out, n_state, dtype=dt)
    gc_part = torch.zeros(batch, n_d_blocks, l_out, n_state, dtype=dt)
    ga_acc = torch.zeros(batch, d_model, n_state, dtype=dt)
    gd_acc = torch.zeros(batch, d_model, dtype=dt)
    gw_acc = torch.zeros(batch, d_model, conv_k, dtype=dt)
    gcb_acc = torch.zeros(batch, d_model, dtype=dt)

    ag_carry = torch.zeros(batch, d_model, n_state, dtype=dt)
    h_cur = torch.empty(batch, d_model, n_state, dtype=dt)
    hbuf = torch.empty(chunk_k, batch, d_model, n_state, dtype=dt)
    zbuf = torch.empty(chunk_k, batch, d_model, dtype=dt)
    cvbuf = torch.empty(chunk_k, batch, d_model, dtype=dt)
    d_splits = [slice(d0, d0 + block_d) for d0 in range(0, d_model, block_d)]

    for ci in range(n_chunks):
        c = n_chunks - 1 - ci
        t0 = c * chunk_k
        h_prev = ckpts[c].clone()
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


class TestBwdKernelReplicaParity:
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
        bound: float = 1e-5,
    ) -> None:
        args = _inputs(b, l_out, d, n, k, seed)
        if saturate:
            args[3].view(-1)[::7] = 25.0
            args[3].view(-1)[::11] = 95.0
        got = _bwd_pipeline_replica(*args, 1e-5, chunk_k, block_d, block_t)
        want = _fused_bwd_eager(*args, eps=1e-5)
        for field, g, r in zip(FusedBlockGrads._fields, got, want, strict=True):
            max_err = (g - r).abs().max().item()
            scale = r.abs().max().clamp(min=1.0).item()
            assert max_err / scale < bound, f"{field} diverges: scale_rel={max_err / scale:.3e}"

    def test_standard_window(self) -> None:
        self._check(2, 64, 12, 16, 4)

    def test_window_two(self) -> None:
        self._check(2, 48, 8, 16, 2, chunk_k=16)

    def test_window_one_degenerate(self) -> None:
        self._check(1, 32, 6, 8, 1)

    def test_window_at_budget(self) -> None:
        # K=8 is MAX_CONV_K, the widest window the launcher admits.
        self._check(1, 32, 6, 8, 8)

    def test_single_chunk_single_step(self) -> None:
        self._check(1, 1, 8, 16, 4, chunk_k=1)

    def test_chunk_one_many_chunks(self) -> None:
        self._check(2, 16, 8, 8, 4, chunk_k=1)

    def test_d_block_tail(self) -> None:
        # d_model=10, block_d=4: masked tail lanes in grad_B/grad_C partials and norm loops.
        self._check(2, 32, 10, 8, 4, block_d=4)

    def test_t_block_tail(self) -> None:
        # l_out=24 with block_t=16: the norm kernel's masked t-rows.
        self._check(1, 24, 8, 8, 4, block_t=16)

    def test_saturated_softplus_branch(self) -> None:
        self._check(2, 32, 8, 8, 4, saturate=True)

    def test_long_sequence_decay(self) -> None:
        self._check(1, 512, 6, 16, 4, seed=7, chunk_k=16, bound=1e-4)

    def test_fp64_tight(self) -> None:
        args = _inputs(2, 64, 12, 16, 4, seed=11, dtype=torch.float64)
        got = _bwd_pipeline_replica(*args, 1e-5, 8, 4, 8)
        want = _fused_bwd_eager(*args, eps=1e-5)
        for field, g, r in zip(FusedBlockGrads._fields, got, want, strict=True):
            max_err = (g - r).abs().max().item()
            assert max_err < 1e-12, f"fp64 {field} diverges: {max_err:.3e}"


class TestBwdReplicaNonFinites:
    """Mask parity: the sdw factoring and recompute path must match autograd's NaN/Inf pattern."""

    def _masks_match(self, args: tuple[torch.Tensor, ...]) -> None:
        got = _bwd_pipeline_replica(*args, 1e-5, 8, 4, 8)
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

    def test_nan_in_x(self) -> None:
        args = list(_inputs(1, 16, 6, 8, 4, seed=4))
        args[0][0, 9, 3] = float("nan")
        self._masks_match(tuple(args))

    @pytest.mark.parametrize("val", [float("inf"), float("-inf")])
    def test_inf_in_x(self, val: float) -> None:
        args = list(_inputs(1, 16, 6, 8, 4, seed=4))
        args[0][0, 9, 3] = val
        self._masks_match(tuple(args))
