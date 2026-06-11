"""CPU tests for the C6 fused-block backward op (all nine gradients)."""

from __future__ import annotations

import math

import pytest
import torch

from flash_mamba_rl.kernels.ops import fused_block_backward, fused_block_forward
from flash_mamba_rl.kernels.references.fused_block_backward import (
    FusedBlockGrads,
    reference_fused_block_backward,
)


def _fused_inputs(
    b: int,
    l_out: int,
    d: int,
    n: int,
    k: int = 4,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    x = torch.randn(b, l_out + k - 1, d).to(dtype)
    conv_w = (torch.randn(d, 1, k) / math.sqrt(k)).to(dtype)
    conv_b = (0.5 * torch.randn(d)).to(dtype)
    delta = torch.randn(b, l_out, d).to(dtype)
    a = (-torch.rand(d, n)).to(dtype)
    b_proj = torch.randn(b, l_out, n).to(dtype)
    c_proj = torch.randn(b, l_out, n).to(dtype)
    d_skip = torch.randn(d).to(dtype)
    norm_w = (1.0 + 0.25 * torch.randn(d)).to(dtype)
    dy = torch.randn(b, l_out, d).to(dtype)
    return x, conv_w, conv_b, delta, a, b_proj, c_proj, d_skip, norm_w, dy


class TestFusedBlockBackwardCpu:
    @pytest.mark.parametrize(
        ("b", "l_out", "d", "n", "k"),
        [
            (1, 8, 4, 8, 4),
            (2, 32, 12, 16, 4),
            (2, 16, 96, 16, 2),
            (1, 24, 7, 10, 3),
            (1, 8, 4, 8, 1),
        ],
    )
    def test_fp32_cpu_bitwise_equals_reference(
        self, b: int, l_out: int, d: int, n: int, k: int
    ) -> None:
        args = _fused_inputs(b, l_out, d, n, k=k)
        ours = fused_block_backward(*args, conv_kernel_size=k, chunk_size=8)
        ref = reference_fused_block_backward(*args, conv_kernel_size=k, chunk_size=8)
        for field, got, want in zip(FusedBlockGrads._fields, ours, ref, strict=True):
            assert torch.equal(got, want), f"{field}: eager path must replicate the reference"

    def test_grad_shapes_match_inputs(self) -> None:
        args = _fused_inputs(2, 16, 8, 8)
        grads = fused_block_backward(*args, chunk_size=8)
        for got, inp in zip(grads, args[:9], strict=True):
            assert got.shape == inp.shape

    def test_chunk_size_validation(self) -> None:
        args = _fused_inputs(1, 10, 4, 8)
        with pytest.raises(ValueError, match="divisible"):
            fused_block_backward(*args, chunk_size=8)

    def test_conv_kernel_size_mismatch_rejected(self) -> None:
        args = _fused_inputs(1, 8, 4, 8, k=4)
        with pytest.raises(ValueError, match="disagrees"):
            fused_block_backward(*args, conv_kernel_size=3, chunk_size=8)

    def test_fp64_stays_fp64(self) -> None:
        args = tuple(t.to(torch.float64) for t in _fused_inputs(1, 8, 4, 8))
        grads = fused_block_backward(*args, chunk_size=8)
        for field, g in zip(FusedBlockGrads._fields, grads, strict=True):
            assert g.dtype == torch.float64, field
            assert torch.isfinite(g).all(), field

    def test_half_dtypes_round_once(self) -> None:
        for dtype in (torch.float16, torch.bfloat16):
            args16 = tuple(t.to(dtype) for t in _fused_inputs(2, 16, 8, 8))
            grads = fused_block_backward(*args16, chunk_size=8)
            ref = reference_fused_block_backward(
                *(t.to(torch.float32) for t in args16), chunk_size=8
            )
            for field, got, want in zip(FusedBlockGrads._fields, grads, ref, strict=True):
                assert got.dtype == dtype, field
                assert torch.equal(got, want.to(dtype)), field

    def test_consistent_with_public_forward_autograd(self) -> None:
        # The op pair must be self-consistent: differentiating the public
        # forward must give the same gradients the public backward returns.
        args = _fused_inputs(2, 8, 6, 8, seed=11)
        direct = fused_block_backward(*args, chunk_size=8)

        leaves = tuple(t.detach().requires_grad_(True) for t in args[:9])
        y = fused_block_forward(*leaves, chunk_size=8)
        via_autograd = torch.autograd.grad(y, leaves, args[9])
        for field, got, want in zip(FusedBlockGrads._fields, direct, via_autograd, strict=True):
            assert torch.equal(got, want), field

    def test_differentiable_wrt_dy_when_required(self) -> None:
        # CMP-02's gradcheck differentiates the op w.r.t. dy; the eager path
        # must build that graph (the VJP is linear in dy).
        args = list(_fused_inputs(1, 8, 4, 8))
        args[9] = args[9].requires_grad_(True)
        grads = fused_block_backward(*args, chunk_size=8)
        assert grads.grad_x.requires_grad
        (dd,) = torch.autograd.grad(grads.grad_x.sum(), args[9])
        assert dd.shape == args[9].shape
        assert torch.isfinite(dd).all()

    def test_gradcheck_wrt_dy_fp64(self) -> None:
        args = tuple(t.to(torch.float64) for t in _fused_inputs(1, 4, 3, 4, k=2))
        dy = args[9].clone().requires_grad_(True)
        assert torch.autograd.gradcheck(
            lambda g: (
                fused_block_backward(*args[:9], g, conv_kernel_size=2, chunk_size=4).grad_delta
            ),
            (dy,),
            eps=1e-6,
            atol=1e-8,
        )

    def test_nonfinite_dy_masks_match_reference(self) -> None:
        args = list(_fused_inputs(1, 16, 6, 8, seed=5))
        args[9][0, 7, 2] = float("nan")
        args[9][0, 3, 1] = float("inf")
        ours = fused_block_backward(*args, chunk_size=8)
        ref = reference_fused_block_backward(*args, chunk_size=8)
        for field, got, want in zip(FusedBlockGrads._fields, ours, ref, strict=True):
            assert torch.equal(got.isnan(), want.isnan()), field
            assert torch.equal(got.isinf(), want.isinf()), field
