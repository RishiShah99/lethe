"""Tests for the Mamba-3 PyTorch reference implementations."""

import pytest
import torch

from lethe.kernels.references import (
    FusedBlockGrads,
    SelectiveScanGrads,
    reference_backward_selective_scan,
    reference_complex_scan_rope,
    reference_forward_chunked_scan,
    reference_fused_block_backward,
    reference_fused_block_forward,
    reference_mimo_backward,
    reference_mimo_forward,
)

B = 2  # batch
L = 8  # sequence length
D = 4  # model dim
N = 8  # state dim
CHUNK = 4


def _make_scan_inputs(
    b: int = B, seq: int = L, d: int = D, n: int = N
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return (u, delta, A, B_proj, C_proj, D_skip) for a SISO scan."""
    torch.manual_seed(0)
    u = torch.randn(b, seq, d)
    delta = torch.randn(b, seq, d)
    A = -torch.rand(d, n)  # negative for stability
    B_proj = torch.randn(b, seq, n)
    C_proj = torch.randn(b, seq, n)
    D_skip = torch.randn(d)
    return u, delta, A, B_proj, C_proj, D_skip


class TestForwardChunkedScan:
    def test_import(self) -> None:
        assert callable(reference_forward_chunked_scan)

    def test_output_shape(self) -> None:
        u, delta, A, B_p, C_p, D_s = _make_scan_inputs()
        y = reference_forward_chunked_scan(u, delta, A, B_p, C_p, D_s, chunk_size=CHUNK)
        assert y.shape == (B, L, D)

    def test_no_nan_inf(self) -> None:
        u, delta, A, B_p, C_p, D_s = _make_scan_inputs()
        y = reference_forward_chunked_scan(u, delta, A, B_p, C_p, D_s, chunk_size=CHUNK)
        assert torch.isfinite(y).all(), "NaN or Inf in forward scan output"

    def test_dtype_preserved(self) -> None:
        u, delta, A, B_p, C_p, D_s = _make_scan_inputs()
        y = reference_forward_chunked_scan(u, delta, A, B_p, C_p, D_s, chunk_size=CHUNK)
        assert y.dtype == torch.float32

    def test_invalid_seq_len_raises(self) -> None:
        # L=8 with chunk_size=3, not divisible
        u, delta, A, B_p, C_p, D_s = _make_scan_inputs(seq=8)
        with pytest.raises(ValueError, match="divisible"):
            reference_forward_chunked_scan(u, delta, A, B_p, C_p, D_s, chunk_size=3)

    def test_chunk_size_one(self) -> None:
        """chunk_size=1 should behave identically to default chunking."""
        u, delta, A, B_p, C_p, D_s = _make_scan_inputs()
        y1 = reference_forward_chunked_scan(u, delta, A, B_p, C_p, D_s, chunk_size=1)
        y4 = reference_forward_chunked_scan(u, delta, A, B_p, C_p, D_s, chunk_size=4)
        assert torch.allclose(y1, y4, atol=1e-5), "chunk_size=1 and 4 disagree"

    def test_chunk_size_full_sequence(self) -> None:
        u, delta, A, B_p, C_p, D_s = _make_scan_inputs()
        y = reference_forward_chunked_scan(u, delta, A, B_p, C_p, D_s, chunk_size=L)
        assert y.shape == (B, L, D)


class TestBackwardSelectiveScan:
    def test_import(self) -> None:
        assert callable(reference_backward_selective_scan)

    def test_returns_named_tuple(self) -> None:
        u, delta, A, B_p, C_p, D_s = _make_scan_inputs()
        dy = torch.ones(B, L, D)
        result = reference_backward_selective_scan(u, delta, A, B_p, C_p, D_s, dy, chunk_size=CHUNK)
        assert isinstance(result, SelectiveScanGrads)

    def test_grad_shapes(self) -> None:
        u, delta, A, B_p, C_p, D_s = _make_scan_inputs()
        dy = torch.ones(B, L, D)
        g = reference_backward_selective_scan(u, delta, A, B_p, C_p, D_s, dy, chunk_size=CHUNK)
        assert g.grad_u.shape == u.shape
        assert g.grad_delta.shape == delta.shape
        assert g.grad_A.shape == A.shape
        assert g.grad_B.shape == B_p.shape
        assert g.grad_C.shape == C_p.shape
        assert g.grad_D.shape == D_s.shape

    def test_no_nan_inf_in_grads(self) -> None:
        u, delta, A, B_p, C_p, D_s = _make_scan_inputs()
        dy = torch.ones(B, L, D)
        g = reference_backward_selective_scan(u, delta, A, B_p, C_p, D_s, dy, chunk_size=CHUNK)
        for field in g._fields:
            tensor = getattr(g, field)
            assert torch.isfinite(tensor).all(), f"NaN/Inf in {field}"

    def test_grads_nonzero(self) -> None:
        u, delta, A, B_p, C_p, D_s = _make_scan_inputs()
        dy = torch.ones(B, L, D)
        g = reference_backward_selective_scan(u, delta, A, B_p, C_p, D_s, dy, chunk_size=CHUNK)
        for field in g._fields:
            tensor = getattr(g, field)
            assert tensor.abs().sum() > 0, f"All-zero gradient for {field}"


class TestMimoBackwardSmoke:
    def test_import(self) -> None:
        assert callable(reference_mimo_backward)

    def test_forward_import(self) -> None:
        assert callable(reference_mimo_forward)

    def test_forward_output_shape(self) -> None:
        """Quick smoke: forward produces the right output shape."""
        torch.manual_seed(0)
        R = 2
        x = torch.randn(B, L, D, N)
        _B = torch.randn(B, L, R, D, N)
        _C = torch.randn(B, L, R, D, N)
        dt = torch.rand(B, L, D) * 0.5 + 0.1
        alpha = torch.rand(B, L, D) * 0.5 + 0.4
        mimo_x = torch.ones(D, R, N)
        mimo_o = torch.ones(D, R, N)
        y = reference_mimo_forward(x, _B, _C, dt, alpha, mimo_x, mimo_o)
        assert y.shape == (B, L, D, N)


class TestComplexScanRopeSmoke:
    def test_import(self) -> None:
        assert callable(reference_complex_scan_rope)

    def test_output_shape(self) -> None:
        """Quick smoke: forward produces the right output shape."""
        torch.manual_seed(0)
        num_rope = N // 4  # d_state=8; num_rope=2; rotary_dim=4 <= 8
        x = torch.randn(B, L, D, N)
        _B = torch.randn(B, L, D, N * 2)
        _C = torch.randn(B, L, D, N * 2)
        dt = torch.rand(B, L, D) * 0.5 + 0.1
        A = -torch.rand(D) - 0.1
        angle_proj = torch.randn(B, L, D, num_rope)
        y = reference_complex_scan_rope(x, _B, _C, dt, A, angle_proj)
        assert y.shape == (B, L, D, N)


CONV_K = 4
# Input length is L_out + (K-1); L_out must be divisible by CHUNK
L_OUT = 8
L_IN = L_OUT + CONV_K - 1  # = 11


def _make_fused_inputs(
    b: int = B,
    l_in: int = L_IN,
    l_out: int = L_OUT,
    d: int = D,
    n: int = N,
    k: int = CONV_K,
) -> tuple[
    torch.Tensor,  # x
    torch.Tensor,  # conv_weight
    torch.Tensor,  # conv_bias
    torch.Tensor,  # delta
    torch.Tensor,  # A
    torch.Tensor,  # B_proj
    torch.Tensor,  # C_proj
    torch.Tensor,  # D_skip
    torch.Tensor,  # norm_weight
]:
    torch.manual_seed(1)
    x = torch.randn(b, l_in, d)
    conv_weight = torch.randn(d, 1, k) * 0.1
    conv_bias = torch.zeros(d)
    delta = torch.randn(b, l_out, d)
    A = -torch.rand(d, n)
    B_p = torch.randn(b, l_out, n)
    C_p = torch.randn(b, l_out, n)
    D_s = torch.randn(d)
    norm_weight = torch.ones(d)
    return x, conv_weight, conv_bias, delta, A, B_p, C_p, D_s, norm_weight


class TestFusedBlockForward:
    def test_import(self) -> None:
        assert callable(reference_fused_block_forward)

    def test_output_shape(self) -> None:
        args = _make_fused_inputs()
        y = reference_fused_block_forward(*args, conv_kernel_size=CONV_K, chunk_size=CHUNK)
        assert y.shape == (B, L_OUT, D)

    def test_no_nan_inf(self) -> None:
        args = _make_fused_inputs()
        y = reference_fused_block_forward(*args, conv_kernel_size=CONV_K, chunk_size=CHUNK)
        assert torch.isfinite(y).all(), "NaN or Inf in fused block forward output"

    def test_dtype_preserved(self) -> None:
        args = _make_fused_inputs()
        y = reference_fused_block_forward(*args, conv_kernel_size=CONV_K, chunk_size=CHUNK)
        assert y.dtype == torch.float32

    def test_norm_weight_scales_output(self) -> None:
        """Doubling norm_weight should double the output."""
        args = list(_make_fused_inputs())
        y1 = reference_fused_block_forward(*args, conv_kernel_size=CONV_K, chunk_size=CHUNK)
        args[8] = args[8] * 2.0  # norm_weight is index 8
        y2 = reference_fused_block_forward(*args, conv_kernel_size=CONV_K, chunk_size=CHUNK)
        assert torch.allclose(y2, y1 * 2.0, atol=1e-5)


class TestFusedBlockBackward:
    def test_import(self) -> None:
        assert callable(reference_fused_block_backward)

    def test_returns_named_tuple(self) -> None:
        args = _make_fused_inputs()
        dy = torch.ones(B, L_OUT, D)
        result = reference_fused_block_backward(
            *args, dy, conv_kernel_size=CONV_K, chunk_size=CHUNK
        )
        assert isinstance(result, FusedBlockGrads)

    def test_grad_shapes(self) -> None:
        args = _make_fused_inputs()
        x, conv_w, conv_b, delta, A, B_p, C_p, D_s, nw = args
        dy = torch.ones(B, L_OUT, D)
        g = reference_fused_block_backward(*args, dy, conv_kernel_size=CONV_K, chunk_size=CHUNK)
        assert g.grad_x.shape == x.shape
        assert g.grad_conv_weight.shape == conv_w.shape
        assert g.grad_conv_bias.shape == conv_b.shape
        assert g.grad_delta.shape == delta.shape
        assert g.grad_A.shape == A.shape
        assert g.grad_B.shape == B_p.shape
        assert g.grad_C.shape == C_p.shape
        assert g.grad_D.shape == D_s.shape
        assert g.grad_norm_weight.shape == nw.shape

    def test_no_nan_inf_in_grads(self) -> None:
        args = _make_fused_inputs()
        dy = torch.ones(B, L_OUT, D)
        g = reference_fused_block_backward(*args, dy, conv_kernel_size=CONV_K, chunk_size=CHUNK)
        for field in g._fields:
            tensor = getattr(g, field)
            assert torch.isfinite(tensor).all(), f"NaN/Inf in {field}"

    def test_grads_nonzero(self) -> None:
        args = _make_fused_inputs()
        dy = torch.ones(B, L_OUT, D)
        g = reference_fused_block_backward(*args, dy, conv_kernel_size=CONV_K, chunk_size=CHUNK)
        for field in g._fields:
            tensor = getattr(g, field)
            assert tensor.abs().sum() > 0, f"All-zero gradient for {field}"
