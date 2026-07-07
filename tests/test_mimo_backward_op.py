"""CPU-side validation of the C3 MIMO backward op (the eager path).

The Triton path needs the box; everything here pins the eager path's
contract — bitwise oracle parity on fp32, the mixed-precision rounding
contract on half dtypes, gradcheck-ability — so the GPU run only has to
answer GPU-specific questions.
"""

from __future__ import annotations

import torch

from lethe.kernels.ops import mimo_backward
from lethe.kernels.references.mimo_backward import (
    MimoGrads,
    reference_mimo_backward,
    reference_mimo_forward,
)

MIMO_FIELDS = MimoGrads._fields


def _mimo_inputs(
    b: int = 2,
    seq: int = 6,
    rank: int = 2,
    h: int = 2,
    p: int = 3,
    n: int = 4,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    x = torch.randn(b, seq, h, p, dtype=dtype)
    bb = torch.randn(b, seq, rank, h, n, dtype=dtype)
    cc = torch.randn(b, seq, rank, h, n, dtype=dtype)
    dt = torch.rand(b, seq, h, dtype=dtype) * 0.1 + 1e-3
    alpha = torch.exp(-dt * torch.rand(h, dtype=dtype))
    mimo_x = torch.full((h, rank, p), 1.0 / rank, dtype=dtype)
    mimo_x += torch.randn(h, rank, p, dtype=dtype) * 0.1
    mimo_o = torch.full((h, rank, p), 1.0 / rank, dtype=dtype)
    mimo_o += torch.randn(h, rank, p, dtype=dtype) * 0.1
    return x, bb, cc, dt, alpha, mimo_x, mimo_o


class TestMimoBackwardCpu:
    def test_matches_reference_fp32(self) -> None:
        args = _mimo_inputs()
        dy = torch.randn(2, 6, 2, 3)
        ours = mimo_backward(*args, dy)
        ref = reference_mimo_backward(*args, dy)
        for field, got, want in zip(MIMO_FIELDS, ours, ref, strict=True):
            assert torch.equal(got, want), f"{field}: eager path must replicate the reference"

    def test_fp16_grads_close_to_fp32_oracle(self) -> None:
        args32 = _mimo_inputs(seq=8, p=4)
        args16 = tuple(t.to(torch.float16) for t in args32)
        dy16 = torch.randn(2, 8, 2, 4, dtype=torch.float16)
        ours = mimo_backward(*args16, dy16)
        ref = reference_mimo_backward(
            *(t.to(torch.float32) for t in args16), dy16.to(torch.float32)
        )
        for field, got, want in zip(MIMO_FIELDS, ours, ref, strict=True):
            assert got.dtype == torch.float16, field
            assert torch.allclose(got.float(), want, atol=1e-2, rtol=1e-2), field

    def test_bf16_dtype_preserved(self) -> None:
        args = tuple(t.to(torch.bfloat16) for t in _mimo_inputs())
        dy = torch.randn(2, 6, 2, 3, dtype=torch.bfloat16)
        grads = mimo_backward(*args, dy)
        for field, g in zip(MIMO_FIELDS, grads, strict=True):
            assert g.dtype == torch.bfloat16, field
            assert torch.isfinite(g.float()).all(), field

    def test_fp64_native(self) -> None:
        args64 = tuple(t.to(torch.float64) for t in _mimo_inputs())
        dy = torch.randn(2, 6, 2, 3, dtype=torch.float64)
        grads = mimo_backward(*args64, dy)
        ref = reference_mimo_backward(*args64, dy)
        for field, got, want in zip(MIMO_FIELDS, grads, ref, strict=True):
            assert got.dtype == torch.float64, field
            assert torch.equal(got, want), field

    def test_grad_shapes_match_inputs(self) -> None:
        args = _mimo_inputs(b=1, seq=4, rank=3, h=2, p=5, n=4)
        dy = torch.randn(1, 4, 2, 5)
        grads = mimo_backward(*args, dy)
        for g, t in zip(grads, args, strict=True):
            assert g.shape == t.shape

    def test_consistent_with_reference_forward_autograd(self) -> None:
        args = _mimo_inputs(seed=11)
        dy = torch.randn(2, 6, 2, 3)
        direct = mimo_backward(*args, dy)

        leaves = tuple(t.detach().requires_grad_(True) for t in args)
        y = reference_mimo_forward(*leaves)
        via_autograd = torch.autograd.grad(y, leaves, dy)
        for field, got, want in zip(MIMO_FIELDS, direct, via_autograd, strict=True):
            assert torch.equal(got, want), field

    def test_differentiable_wrt_dy_when_required(self) -> None:
        # CMP-02's gradcheck differentiates the op w.r.t. dy; the eager path
        # must build that graph (the VJP is linear in dy).
        args = _mimo_inputs()
        dy = torch.randn(2, 6, 2, 3, requires_grad=True)
        grads = mimo_backward(*args, dy)
        assert grads.grad_x.requires_grad
        (dd,) = torch.autograd.grad(grads.grad_x.sum(), dy)
        assert dd.shape == dy.shape
        assert torch.isfinite(dd).all()

    def test_gradcheck_fp64_wrt_dy(self) -> None:
        args = tuple(t.to(torch.float64) for t in _mimo_inputs(b=1, seq=3, rank=2, h=1, p=2, n=2))
        dy = torch.randn(1, 3, 1, 2, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            lambda g: mimo_backward(*args, g).grad_B, (dy,), eps=1e-6, atol=1e-4
        )

    def test_nonfinite_dy_masks_match_oracle(self) -> None:
        # EXC-01 flows non-finites through the primary (dy); the eager path
        # must mint NaN/Inf exactly where the oracle does.
        args = _mimo_inputs(seed=5)
        dy = torch.randn(2, 6, 2, 3)
        dy[0, 1, 0, 0] = float("inf")
        dy[1, 3, 1, 2] = float("nan")
        ours = mimo_backward(*args, dy)
        ref = reference_mimo_backward(*args, dy)
        for field, got, want in zip(MIMO_FIELDS, ours, ref, strict=True):
            assert torch.equal(torch.isnan(got), torch.isnan(want)), field
            assert torch.equal(torch.isinf(got), torch.isinf(want)), field
