"""GPU validation for the hand-written Phase C kernels.

Runs on the fleet box (``bash scratch/detach.sh bash scratch/c1_gpu_suite.sh``).
Skips cleanly on CPU-only hosts. The headline test is
``TestC1ContractGates::test_all_gates_pass_on_cuda`` — C1's 12/12 exit
criterion — with parity-vs-official as the independent oracle check.
"""

from __future__ import annotations

import pytest
import torch

from flash_mamba_rl.kernels.ops import forward_chunked_scan, triton_scan_resource_meta
from flash_mamba_rl.kernels.references import reference_forward_chunked_scan
from flash_mamba_rl.verifier.op_harness import verify_scan_op
from flash_mamba_rl.verifier.timing import benchmark

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    _HAS_MAMBA = True
except ImportError:
    _HAS_MAMBA = False


def _scan_inputs(
    b: int,
    seq: int,
    d: int,
    n: int,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    u = torch.randn(b, seq, d, device=dev).to(dtype)
    delta = torch.randn(b, seq, d, device=dev).to(dtype)
    a = (-torch.rand(d, n, device=dev)).to(dtype)
    b_proj = torch.randn(b, seq, n, device=dev).to(dtype)
    c_proj = torch.randn(b, seq, n, device=dev).to(dtype)
    d_skip = torch.randn(d, device=dev).to(dtype)
    return u, delta, a, b_proj, c_proj, d_skip


@pytest.mark.gpu
@requires_gpu
class TestC1TritonParity:
    @pytest.mark.parametrize(
        ("b", "seq", "d", "n"),
        [
            (1, 8, 4, 8),  # tiny
            (2, 64, 96, 16),  # non-pow2 D
            (3, 128, 100, 10),  # non-pow2 D and N
            (2, 256, 512, 32),  # larger state
        ],
    )
    def test_fp32_matches_reference(self, b: int, seq: int, d: int, n: int) -> None:
        args = _scan_inputs(b, seq, d, n)
        y_triton = forward_chunked_scan(*args, chunk_size=8)
        y_ref = reference_forward_chunked_scan(*args, chunk_size=8)
        max_err = (y_triton - y_ref).abs().max().item()
        # Same standard as the official-kernel parity test: fp32 reorder
        # noise grows ~eps*sqrt(L)*|y| (measured 2.6e-4 at L=256, N=32).
        assert torch.allclose(y_triton, y_ref, atol=1e-4, rtol=1e-3), f"max_err={max_err:.3e}"

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_low_precision_matches_fp32_oracle(self, dtype: torch.dtype) -> None:
        args = _scan_inputs(2, 128, 64, 16, dtype=dtype)
        y = forward_chunked_scan(*args, chunk_size=8)
        y_ref = reference_forward_chunked_scan(*(t.to(torch.float32) for t in args), chunk_size=8)
        assert y.dtype == dtype
        atol = 1e-3 if dtype == torch.float16 else 1e-2
        assert torch.allclose(y.float(), y_ref, atol=atol, rtol=atol)

    def test_deterministic_across_runs(self) -> None:
        args = _scan_inputs(2, 128, 64, 16)
        first = forward_chunked_scan(*args, chunk_size=8)
        for _ in range(4):
            assert torch.equal(forward_chunked_scan(*args, chunk_size=8), first)

    def test_backward_matches_reference_autograd(self) -> None:
        args_ours = _scan_inputs(2, 32, 16, 8, seed=5)
        args_ref = tuple(t.clone() for t in args_ours)
        for t in (*args_ours, *args_ref):
            t.requires_grad_(True)
        grad_out = torch.randn(2, 32, 16, device="cuda")
        forward_chunked_scan(*args_ours, chunk_size=8).backward(grad_out)
        reference_forward_chunked_scan(*args_ref, chunk_size=8).backward(grad_out)
        for ours, ref in zip(args_ours, args_ref, strict=True):
            assert ours.grad is not None and ref.grad is not None
            assert torch.allclose(ours.grad, ref.grad, atol=1e-4, rtol=1e-4)


@pytest.mark.gpu
@requires_gpu
class TestC1ContractGates:
    def test_all_gates_pass_on_cuda(self) -> None:
        # Warm-up launch so the resource meta reflects a compiled kernel.
        warm = _scan_inputs(1, 8, 4, 16)
        forward_chunked_scan(*warm, chunk_size=8)
        meta = triton_scan_resource_meta()

        results = verify_scan_op(forward_chunked_scan, device="cuda", resource_meta=meta)
        failed = {
            name: (result.reason, result.details)
            for name, result in results.items()
            if not result.passed
        }
        assert not failed, f"gates failed on CUDA (resource_meta={meta}): {failed}"

    def test_resource_meta_extracted(self) -> None:
        warm = _scan_inputs(1, 8, 4, 16)
        forward_chunked_scan(*warm, chunk_size=8)
        meta = triton_scan_resource_meta()
        assert meta is not None, "no resource metadata from the compiled kernel cache"
        assert 0 < meta["n_regs"] <= 255


@pytest.mark.gpu
@requires_gpu
@pytest.mark.skipif(not _HAS_MAMBA, reason="mamba_ssm not installed")
class TestC1VsOfficialMamba:
    def test_matches_selective_scan_fn(self) -> None:
        # Independent oracle: the official Mamba-1 CUDA kernel computes the
        # same SISO recurrence in [B, D, L] layout with delta_softplus=True.
        u, delta, a, b_proj, c_proj, d_skip = _scan_inputs(2, 256, 64, 16, seed=7)
        y_ours = forward_chunked_scan(u, delta, a, b_proj, c_proj, d_skip, chunk_size=8)

        y_official = selective_scan_fn(
            u.transpose(1, 2).contiguous(),
            delta.transpose(1, 2).contiguous(),
            a,
            b_proj.transpose(1, 2).unsqueeze(1).contiguous(),  # [B, 1, N, L]
            c_proj.transpose(1, 2).unsqueeze(1).contiguous(),
            d_skip,
            delta_softplus=True,
        ).transpose(1, 2)

        max_err = (y_ours - y_official).abs().max().item()
        assert torch.allclose(y_ours, y_official, atol=1e-4, rtol=1e-4), (
            f"disagrees with official mamba_ssm kernel: max_err={max_err:.3e}"
        )


@pytest.mark.gpu
@pytest.mark.slow
@requires_gpu
class TestC1BenchSmoke:
    def test_faster_than_eager_reference(self) -> None:
        args = _scan_inputs(4, 512, 256, 16)
        t_triton = benchmark(
            lambda: forward_chunked_scan(*args, chunk_size=8), (), warmup=5, trials=20
        )
        t_ref = benchmark(
            lambda: reference_forward_chunked_scan(*args, chunk_size=8), (), warmup=2, trials=5
        )
        assert t_triton.median_ms < t_ref.median_ms, (
            f"triton {t_triton.median_ms:.3f} ms not faster than eager {t_ref.median_ms:.3f} ms"
        )
