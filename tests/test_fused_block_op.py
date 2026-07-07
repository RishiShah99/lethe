"""CPU tests for the C5 fused-block forward op (conv1d + SiLU + scan + RMSNorm)."""

from __future__ import annotations

import math

import pytest
import torch

from lethe.kernels.ops import fused_block_forward
from lethe.kernels.references import reference_fused_block_forward


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
    return x, conv_w, conv_b, delta, a, b_proj, c_proj, d_skip, norm_w


class TestFusedBlockForwardCpu:
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
        ours = fused_block_forward(*args, conv_kernel_size=k, chunk_size=8)
        ref = reference_fused_block_forward(*args, conv_kernel_size=k, chunk_size=8)
        assert torch.equal(ours, ref)

    def test_output_shape_drops_conv_halo(self) -> None:
        args = _fused_inputs(2, 16, 8, 8, k=4)
        y = fused_block_forward(*args, chunk_size=8)
        assert y.shape == (2, 16, 8)

    def test_hand_computed_single_step(self) -> None:
        x0, x1 = 0.5, -1.0
        w0, w1, bias = 0.25, 0.5, 0.1
        beta, a_val, b_val, c_val, d_val, g, eps = -0.3, -0.7, 1.2, 0.8, -0.4, 1.1, 1e-5

        conv = x0 * w0 + x1 * w1 + bias
        z = conv / (1.0 + math.exp(-conv))
        dbar = math.log1p(math.exp(beta))
        h = dbar * z * b_val  # h_0 = exp(dbar*a)*0 + dbar*B*z
        y_scan = h * c_val + d_val * z
        expected = y_scan / math.sqrt(y_scan * y_scan + eps) * g

        y = fused_block_forward(
            torch.tensor([[[x0], [x1]]]),
            torch.tensor([[[w0, w1]]]),
            torch.tensor([bias]),
            torch.tensor([[[beta]]]),
            torch.tensor([[a_val]]),
            torch.tensor([[[b_val]]]),
            torch.tensor([[[c_val]]]),
            torch.tensor([d_val]),
            torch.tensor([g]),
            conv_kernel_size=2,
            eps=eps,
            chunk_size=1,
        )
        assert y.shape == (1, 1, 1)
        assert math.isclose(y.item(), expected, rel_tol=1e-6)

    def test_gradcheck_fp64(self) -> None:
        args = tuple(
            t.to(torch.float64).requires_grad_(True) for t in _fused_inputs(1, 4, 3, 4, k=2)
        )
        assert torch.autograd.gradcheck(
            lambda *a: fused_block_forward(*a, conv_kernel_size=2, chunk_size=4),
            args,
            eps=1e-6,
            atol=1e-8,
        )

    def test_chunk_size_validation(self) -> None:
        args = _fused_inputs(1, 10, 4, 8)
        with pytest.raises(ValueError, match="divisible"):
            fused_block_forward(*args, chunk_size=8)

    def test_conv_kernel_size_mismatch_rejected(self) -> None:
        args = _fused_inputs(1, 8, 4, 8, k=4)
        with pytest.raises(ValueError, match="disagrees"):
            fused_block_forward(*args, conv_kernel_size=3, chunk_size=8)

    def test_channel_mismatch_rejected_op_and_reference(self) -> None:
        # The CMP-03 audit variant: primary x's channel dim is halved
        # (D=16) while the baked conv/scan weights stay at D=32. torch's
        # groups=D conv silently upweights 16->32; both the op and the
        # reference must reject instead, so the audit reclassifies this to
        # na (the reference-inapplicable path) rather than a value mismatch.
        args = list(_fused_inputs(2, 64, 32, 16, k=4))
        args[0] = args[0][..., :16].contiguous()  # x -> (2, 67, 16)
        with pytest.raises(ValueError, match="channel mismatch"):
            fused_block_forward(*args, chunk_size=8)
        with pytest.raises(ValueError, match="channel mismatch"):
            reference_fused_block_forward(*args, chunk_size=8)

    def test_fp64_stays_fp64(self) -> None:
        args = tuple(t.to(torch.float64) for t in _fused_inputs(1, 8, 4, 8))
        y = fused_block_forward(*args, chunk_size=8)
        assert y.dtype == torch.float64

    def test_half_dtypes_round_once(self) -> None:
        for dtype in (torch.float16, torch.bfloat16):
            args16 = tuple(t.to(dtype) for t in _fused_inputs(2, 16, 8, 8))
            y = fused_block_forward(*args16, chunk_size=8)
            assert y.dtype == dtype
            y_ref = reference_fused_block_forward(
                *(t.to(torch.float32) for t in args16), chunk_size=8
            )
            assert torch.equal(y, y_ref.to(dtype))

    def test_backward_matches_reference_autograd(self) -> None:
        args_ours = _fused_inputs(2, 8, 6, 8, seed=5)
        args_ref = tuple(t.clone() for t in args_ours)
        for t in (*args_ours, *args_ref):
            t.requires_grad_(True)
        grad_out = torch.randn(2, 8, 6)
        fused_block_forward(*args_ours, chunk_size=8).backward(grad_out)
        reference_fused_block_forward(*args_ref, chunk_size=8).backward(grad_out)
        for ours, ref in zip(args_ours, args_ref, strict=True):
            assert ours.grad is not None and ref.grad is not None
            assert torch.equal(ours.grad, ref.grad)


class TestFusedBlockNonFinites:
    """Pin the contract the Triton kernel must reproduce on GPU.

    On CPU the op is bitwise-equal to the reference, so equality is
    trivial; the value of these tests is pinning the *expected* smear
    semantics (conv window K, then row-wide poisoning through RMSNorm).
    """

    def test_nan_smears_causally_over_conv_window(self) -> None:
        k = 4
        args = list(_fused_inputs(1, 16, 6, 8, k=k))
        args[0][0, 7, 2] = float("nan")
        y = fused_block_forward(*args, chunk_size=8)
        ref = reference_fused_block_forward(*args, chunk_size=8)
        assert torch.equal(y.isnan(), ref.isnan())
        # x feeds output rows t-(K-1)..t via the valid conv on pre-padded
        # input; the scan then carries the poison to every later row, and
        # RMSNorm spreads it across the whole D row.
        first_hit = 7 - (k - 1)
        assert not y[0, :first_hit].isnan().any()
        assert y[0, first_hit:].isnan().all()

    @pytest.mark.parametrize("val", [float("inf"), float("-inf")])
    def test_inf_masks_match_reference(self, val: float) -> None:
        args = list(_fused_inputs(1, 16, 6, 8))
        args[0][0, 5, 1] = val
        y = fused_block_forward(*args, chunk_size=8)
        ref = reference_fused_block_forward(*args, chunk_size=8)
        assert torch.equal(y.isnan(), ref.isnan())
        assert torch.equal(y.isinf(), ref.isinf())
