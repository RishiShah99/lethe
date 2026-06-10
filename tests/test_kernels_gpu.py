"""GPU validation for the hand-written Phase C kernels.

Runs on the fleet box (``bash scratch/detach.sh bash scratch/c2_gpu_suite.sh``).
Skips cleanly on CPU-only hosts. The headline tests are the per-kernel
contract-gate suites — C1's 12/12 and C2's 6x12/12 exit criteria — with
parity-vs-official as the independent oracle checks, plus C2's
num_warps>=4 compile assertion (the config family where the official
Mamba-3 backward dies on sm_100, #904).
"""

from __future__ import annotations

import pytest
import torch

from flash_mamba_rl.kernels.ops import (
    backward_selective_scan,
    complex_scan_rope,
    forward_chunked_scan,
    mimo_backward,
    triton_bwd_scan_resource_meta,
    triton_complex_rope_resource_meta,
    triton_mimo_bwd_resource_meta,
    triton_scan_resource_meta,
)
from flash_mamba_rl.kernels.references import (
    reference_backward_selective_scan,
    reference_forward_chunked_scan,
)
from flash_mamba_rl.kernels.references.complex_scan_rope import reference_complex_scan_rope
from flash_mamba_rl.kernels.references.mimo_backward import MimoGrads, reference_mimo_backward
from flash_mamba_rl.verifier.op_harness import (
    BWD_GRAD_FIELDS,
    verify_bwd_scan_op_all_grads,
    verify_mimo_bwd_op_all_grads,
    verify_rope_op,
    verify_scan_op,
)
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


def _bwd_inputs(
    b: int,
    seq: int,
    d: int,
    n: int,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    args = _scan_inputs(b, seq, d, n, dtype=dtype, seed=seed)
    torch.manual_seed(seed + 1)
    dy = torch.randn(b, seq, d, device="cuda").to(dtype)
    return args, dy


@pytest.mark.gpu
@requires_gpu
class TestC2TritonParity:
    @pytest.mark.parametrize(
        ("b", "seq", "d", "n"),
        [
            (1, 8, 4, 8),  # tiny
            (2, 64, 96, 16),  # non-pow2 D
            (3, 120, 100, 10),  # non-pow2 L (chunk_k=8), D and N
            (2, 256, 512, 32),  # larger state
        ],
    )
    def test_fp32_grads_match_reference(self, b: int, seq: int, d: int, n: int) -> None:
        args, dy = _bwd_inputs(b, seq, d, n)
        ours = backward_selective_scan(*args, dy, chunk_size=8)
        ref = reference_backward_selective_scan(*args, dy, chunk_size=8)
        for field, got, want in zip(BWD_GRAD_FIELDS, ours, ref, strict=True):
            max_err = (got.float() - want.float()).abs().max().item()
            # Cross-implementation reorder noise; grad_A accumulates over
            # batch*L chains so it carries the widest spread (provisional,
            # tightened from B200 measurement like C1's parity bounds).
            assert torch.allclose(got, want, atol=1e-3, rtol=1e-3), (
                f"{field}: max_err={max_err:.3e}"
            )

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_low_precision_grads_match_fp32_oracle(self, dtype: torch.dtype) -> None:
        args, dy = _bwd_inputs(2, 128, 64, 16, dtype=dtype)
        ours = backward_selective_scan(*args, dy, chunk_size=8)
        ref = reference_backward_selective_scan(
            *(t.to(torch.float32) for t in args), dy.to(torch.float32), chunk_size=8
        )
        atol = 1e-2 if dtype == torch.float16 else 5e-2
        for field, got, want in zip(BWD_GRAD_FIELDS, ours, ref, strict=True):
            assert got.dtype == dtype, field
            assert torch.allclose(got.float(), want, atol=atol, rtol=atol), field

    def test_deterministic_across_runs(self) -> None:
        args, dy = _bwd_inputs(2, 128, 64, 16)
        first = backward_selective_scan(*args, dy, chunk_size=8)
        for _ in range(4):
            again = backward_selective_scan(*args, dy, chunk_size=8)
            for field, a_, b_ in zip(BWD_GRAD_FIELDS, first, again, strict=True):
                assert torch.equal(a_, b_), field

    def test_forward_op_autograd_uses_this_kernel(self) -> None:
        # The C1 forward op's autograd backward now dispatches to the C2
        # kernel: differentiating it must reproduce the public backward op
        # bit-for-bit (same launcher, same inputs, deterministic kernel).
        args, dy = _bwd_inputs(2, 64, 32, 16, seed=5)
        direct = backward_selective_scan(*args, dy, chunk_size=8)
        leaves = tuple(t.detach().requires_grad_(True) for t in args)
        y = forward_chunked_scan(*leaves, chunk_size=8)
        via_autograd = torch.autograd.grad(y, leaves, dy)
        for field, got, want in zip(BWD_GRAD_FIELDS, direct, via_autograd, strict=True):
            assert torch.equal(got, want), field


@pytest.mark.gpu
@requires_gpu
class TestC2NumWarpsCompile:
    def test_compiles_and_matches_at_num_warps_4_and_8(self) -> None:
        # The #904 contrast: the official Mamba-3 Triton backward fails to
        # compile at every num_warps >= 4 config on sm_100 (TMEM budget).
        # Ours carries the recurrence without tl.dot, so the TMEM-promotion
        # pass never engages — these launches raising OutOfResources would
        # falsify the C2 story outright.
        from flash_mamba_rl.kernels.ops import _triton_bwd_scan

        args, dy = _bwd_inputs(2, 256, 128, 16)
        base = _triton_bwd_scan.launch_backward_scan(*args, dy, num_warps=2)
        for warps in (4, 8):
            got = _triton_bwd_scan.launch_backward_scan(*args, dy, num_warps=warps)
            for field, g, want in zip(BWD_GRAD_FIELDS, got, base, strict=True):
                # Reduction trees shift with the warp layout; values must
                # stay within reorder noise of the num_warps=2 run.
                assert torch.allclose(g, want, atol=1e-4, rtol=1e-4), (warps, field)
        meta = triton_bwd_scan_resource_meta()
        assert meta is not None
        assert 0 < meta["n_regs"] <= 255


@pytest.mark.gpu
@requires_gpu
class TestC2ContractGates:
    def test_all_gates_all_grads_pass_on_cuda(self) -> None:
        # C2's exit criterion: 12/12 gates on every gradient view, with the
        # compiled kernel's resource envelope feeding RES-02.
        args, dy = _bwd_inputs(1, 8, 4, 16)
        backward_selective_scan(*args, dy, chunk_size=8)  # warm the cache
        meta = triton_bwd_scan_resource_meta()

        all_results = verify_bwd_scan_op_all_grads(
            backward_selective_scan, device="cuda", resource_meta=meta
        )
        failed = {
            f"{view}.{name}": (result.reason, result.details)
            for view, results in all_results.items()
            for name, result in results.items()
            if not result.passed
        }
        assert not failed, f"gates failed on CUDA (resource_meta={meta}): {failed}"

    def test_resource_meta_extracted(self) -> None:
        args, dy = _bwd_inputs(1, 8, 4, 16)
        backward_selective_scan(*args, dy, chunk_size=8)
        meta = triton_bwd_scan_resource_meta()
        assert meta is not None, "no resource metadata from the compiled kernel cache"
        assert 0 < meta["n_regs"] <= 255


@pytest.mark.gpu
@requires_gpu
@pytest.mark.skipif(not _HAS_MAMBA, reason="mamba_ssm not installed")
class TestC2VsOfficialMamba:
    def test_grads_match_selective_scan_fn(self) -> None:
        # Independent oracle: the official Mamba-1 CUDA backward (healthy on
        # Blackwell, unlike the Mamba-3 Triton path) computes the same VJP in
        # [B, D, L] layout.
        u, delta, a, b_proj, c_proj, d_skip = _scan_inputs(2, 256, 64, 16, seed=7)
        torch.manual_seed(11)
        dy = torch.randn(2, 256, 64, device="cuda")
        ours = backward_selective_scan(u, delta, a, b_proj, c_proj, d_skip, dy, chunk_size=8)

        u_o = u.transpose(1, 2).contiguous().requires_grad_(True)
        delta_o = delta.transpose(1, 2).contiguous().requires_grad_(True)
        a_o = a.clone().requires_grad_(True)
        b_o = b_proj.transpose(1, 2).unsqueeze(1).contiguous().requires_grad_(True)
        c_o = c_proj.transpose(1, 2).unsqueeze(1).contiguous().requires_grad_(True)
        d_o = d_skip.clone().requires_grad_(True)
        y = selective_scan_fn(u_o, delta_o, a_o, b_o, c_o, d_o, delta_softplus=True)
        y.backward(dy.transpose(1, 2).contiguous())

        assert u_o.grad is not None and delta_o.grad is not None and a_o.grad is not None
        assert b_o.grad is not None and c_o.grad is not None and d_o.grad is not None
        official = (
            u_o.grad.transpose(1, 2),
            delta_o.grad.transpose(1, 2),
            a_o.grad,
            b_o.grad.squeeze(1).transpose(1, 2),
            c_o.grad.squeeze(1).transpose(1, 2),
            d_o.grad,
        )
        for field, got, want in zip(BWD_GRAD_FIELDS, ours, official, strict=True):
            max_err = (got.float() - want.float()).abs().max().item()
            assert torch.allclose(got, want, atol=1e-3, rtol=1e-3), (
                f"disagrees with official mamba_ssm backward on {field}: max_err={max_err:.3e}"
            )


@pytest.mark.gpu
@pytest.mark.slow
@requires_gpu
class TestC2BenchSmoke:
    def test_faster_than_eager_reference_backward(self) -> None:
        args, dy = _bwd_inputs(4, 512, 256, 16)
        t_triton = benchmark(
            lambda: backward_selective_scan(*args, dy, chunk_size=8), (), warmup=5, trials=20
        )
        t_ref = benchmark(
            lambda: reference_backward_selective_scan(*args, dy, chunk_size=8),
            (),
            warmup=1,
            trials=3,
        )
        assert t_triton.median_ms < t_ref.median_ms, (
            f"triton {t_triton.median_ms:.3f} ms not faster than eager {t_ref.median_ms:.3f} ms"
        )


MIMO_FIELDS = MimoGrads._fields


def _mimo_inputs(
    b: int,
    seq: int,
    rank: int,
    h: int,
    p: int,
    n: int,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    x = torch.randn(b, seq, h, p, device=dev).to(dtype)
    bb = torch.randn(b, seq, rank, h, n, device=dev).to(dtype)
    cc = torch.randn(b, seq, rank, h, n, device=dev).to(dtype)
    dt = (torch.rand(b, seq, h, device=dev) * 0.1 + 1e-3).to(dtype)
    alpha = torch.exp(-dt.float() * torch.rand(h, device=dev)).to(dtype)
    mimo_x = (1.0 / rank + torch.randn(h, rank, p, device=dev) * 0.1).to(dtype)
    mimo_o = (1.0 / rank + torch.randn(h, rank, p, device=dev) * 0.1).to(dtype)
    dy = torch.randn(b, seq, h, p, device=dev).to(dtype)
    return (x, bb, cc, dt, alpha, mimo_x, mimo_o), dy


@pytest.mark.gpu
@requires_gpu
class TestC3TritonParity:
    @pytest.mark.parametrize(
        ("b", "seq", "rank", "h", "p", "n"),
        [
            (1, 8, 1, 2, 4, 8),  # tiny, R=1 degenerate
            (2, 64, 2, 3, 24, 16),  # non-pow2 H and P
            (3, 120, 4, 4, 100, 10),  # non-pow2 L (chunk_k=8), P, N
            (2, 64, 3, 2, 8, 16),  # non-pow2 R (masked BLOCK_R rows)
            (2, 64, 5, 2, 8, 16),  # R=5 (BLOCK_R=8 row-select)
            (2, 256, 4, 8, 64, 128),  # training-like state, N split across blocks
            (2, 128, 2, 4, 16, 100),  # masked last n-block ON the split path
            (1, 1, 2, 2, 4, 8),  # L=1 (single chunk of one step)
            (1, 2, 2, 2, 4, 8),  # L=2 (one chunk, real h_tm1/ag_carry step)
        ],
    )
    def test_fp32_grads_match_reference(
        self, b: int, seq: int, rank: int, h: int, p: int, n: int
    ) -> None:
        args, dy = _mimo_inputs(b, seq, rank, h, p, n)
        ours = mimo_backward(*args, dy)
        ref = reference_mimo_backward(*args, dy)
        for field, got, want in zip(MIMO_FIELDS, ours, ref, strict=True):
            max_err = (got.float() - want.float()).abs().max().item()
            scale = want.float().abs().max().clamp(min=1.0).item()
            # scale_rel is the C2-honest metric: bare absolute parity
            # misleads at near-integrator scales. Provisional bounds,
            # tightened from B200 measurement like C1/C2's.
            assert torch.allclose(got, want, atol=1e-3 * scale, rtol=1e-3), (
                f"{field}: max_err={max_err:.3e} scale_rel={max_err / scale:.3e}"
            )

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_low_precision_grads_match_fp32_oracle(self, dtype: torch.dtype) -> None:
        args, dy = _mimo_inputs(2, 128, 2, 4, 16, 16, dtype=dtype)
        ours = mimo_backward(*args, dy)
        ref = reference_mimo_backward(*(t.to(torch.float32) for t in args), dy.to(torch.float32))
        atol = 1e-2 if dtype == torch.float16 else 5e-2
        for field, got, want in zip(MIMO_FIELDS, ours, ref, strict=True):
            assert got.dtype == dtype, field
            scale = want.float().abs().max().clamp(min=1.0).item()
            assert torch.allclose(got.float(), want, atol=atol * scale, rtol=atol), field

    def test_deterministic_across_runs(self) -> None:
        args, dy = _mimo_inputs(2, 128, 2, 4, 16, 16)
        first = mimo_backward(*args, dy)
        for _ in range(4):
            again = mimo_backward(*args, dy)
            for field, a_, b_ in zip(MIMO_FIELDS, first, again, strict=True):
                assert torch.equal(a_, b_), field

    def test_noncontiguous_inputs_match_contiguous(self) -> None:
        # Pins the launcher's .contiguous() normalisation: a strided view of
        # the same values must produce byte-identical gradients.
        args, dy = _mimo_inputs(2, 64, 2, 4, 16, 16)
        base = mimo_backward(*args, dy)
        x, b_p, c_p, dt, alpha, mimo_x, mimo_o = args
        x_nc = x.transpose(1, 2).contiguous().transpose(1, 2)
        b_nc = b_p.transpose(1, 2).contiguous().transpose(1, 2)
        dy_nc = dy.transpose(2, 3).contiguous().transpose(2, 3)
        assert not (x_nc.is_contiguous() or b_nc.is_contiguous() or dy_nc.is_contiguous())
        again = mimo_backward(x_nc, b_nc, c_p, dt, alpha, mimo_x, mimo_o, dy_nc)
        for field, a_, b_ in zip(MIMO_FIELDS, base, again, strict=True):
            assert torch.equal(a_, b_), field

    def test_nonfinite_dy_masks_match_oracle(self) -> None:
        # EXC-01's contract on the real kernel: non-finites through dy mint
        # NaN/Inf exactly where the autograd oracle does.
        args, dy = _mimo_inputs(2, 64, 2, 2, 8, 16, seed=5)
        dy[0, 9, 0, 1] = float("inf")
        dy[1, 33, 1, 2] = float("nan")
        ours = mimo_backward(*args, dy)
        ref = reference_mimo_backward(*args, dy)
        for field, got, want in zip(MIMO_FIELDS, ours, ref, strict=True):
            assert torch.equal(torch.isnan(got), torch.isnan(want)), field
            assert torch.equal(torch.isinf(got), torch.isinf(want)), field


@pytest.mark.gpu
@requires_gpu
class TestC3NumWarpsCompile:
    @pytest.mark.parametrize(
        ("b", "seq", "rank", "h", "p", "n"),
        [
            (2, 256, 4, 4, 32, 16),
            (2, 128, 2, 4, 16, 128),  # nNb=2: warp layout x n-split reduction
        ],
    )
    def test_compiles_and_matches_at_num_warps_4_and_8(
        self, b: int, seq: int, rank: int, h: int, p: int, n: int
    ) -> None:
        # Same #904 framing as C2: no tl.dot anywhere, so the TMEM-promotion
        # pass never engages and every warp config must compile on sm_100.
        from flash_mamba_rl.kernels.ops import _triton_mimo_bwd

        args, dy = _mimo_inputs(b, seq, rank, h, p, n)
        base = _triton_mimo_bwd.launch_mimo_backward(*args, dy, num_warps=2)
        for warps in (4, 8):
            got = _triton_mimo_bwd.launch_mimo_backward(*args, dy, num_warps=warps)
            for field, g, want in zip(MIMO_FIELDS, got, base, strict=True):
                assert torch.allclose(g, want, atol=1e-4, rtol=1e-4), (warps, field)
        meta = triton_mimo_bwd_resource_meta()
        assert meta is not None
        assert 0 < meta["n_regs"] <= 255


@pytest.mark.gpu
@requires_gpu
class TestC3ContractGates:
    def test_all_gates_all_grads_pass_on_cuda(self) -> None:
        # C3's exit criterion: 12/12 gates on every gradient view, with the
        # compiled kernel's resource envelope feeding RES-02.
        args, dy = _mimo_inputs(1, 8, 4, 1, 4, 16)
        mimo_backward(*args, dy)  # warm the cache
        meta = triton_mimo_bwd_resource_meta()

        all_results = verify_mimo_bwd_op_all_grads(mimo_backward, device="cuda", resource_meta=meta)
        failed = {
            f"{view}.{name}": (result.reason, result.details)
            for view, results in all_results.items()
            for name, result in results.items()
            if not result.passed
        }
        assert not failed, f"gates failed on CUDA (resource_meta={meta}): {failed}"

    def test_resource_meta_extracted(self) -> None:
        args, dy = _mimo_inputs(1, 8, 2, 1, 4, 16)
        mimo_backward(*args, dy)
        meta = triton_mimo_bwd_resource_meta()
        assert meta is not None, "no resource metadata from the compiled kernel cache"
        assert 0 < meta["n_regs"] <= 255


def _rope_inputs(
    b: int,
    seq: int,
    h: int,
    p: int,
    n: int,
    s: int,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    x = torch.randn(b, seq, h, p, device=dev).to(dtype)
    bb = torch.randn(b, seq, h, n, device=dev).to(dtype)
    cc = torch.randn(b, seq, h, n, device=dev).to(dtype)
    dt = (torch.rand(b, seq, h, device=dev) * 0.1 + 1e-3).to(dtype)
    a = (-torch.rand(h, device=dev)).to(dtype)
    angle = torch.randn(b, seq, h, s, device=dev).to(dtype)
    return x, bb, cc, dt, a, angle


@pytest.mark.gpu
@requires_gpu
class TestC4TritonParity:
    @pytest.mark.parametrize(
        ("b", "seq", "h", "p", "n", "s"),
        [
            (1, 8, 2, 4, 8, 3),  # tiny, partial rotary (6 < 8)
            (2, 64, 3, 24, 16, 8),  # non-pow2 H and P, full rotary (16 = N)
            (3, 121, 4, 100, 10, 4),  # odd L, non-pow2 P/N, masked N lanes
            (2, 256, 8, 64, 128, 64),  # training-like, N at MAX_BLOCK_N
            (2, 64, 2, 80, 16, 6),  # P split across blocks (BLOCK_P=64)
            (1, 1, 2, 4, 8, 2),  # L=1
        ],
    )
    def test_fp32_matches_reference(self, b: int, seq: int, h: int, p: int, n: int, s: int) -> None:
        args = _rope_inputs(b, seq, h, p, n, s)
        got = complex_scan_rope(*args)
        want = reference_complex_scan_rope(*args)
        max_err = (got - want).abs().max().item()
        scale = want.abs().max().clamp(min=1.0).item()
        # scale_rel is the C2-honest metric; provisional bounds, tightened
        # from B200 measurement like C1-C3's.
        assert torch.allclose(got, want, atol=1e-3 * scale, rtol=1e-3), (
            f"max_err={max_err:.3e} scale_rel={max_err / scale:.3e}"
        )

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_low_precision_matches_fp32_oracle(self, dtype: torch.dtype) -> None:
        args = _rope_inputs(2, 128, 2, 16, 16, 4, dtype=dtype)
        got = complex_scan_rope(*args)
        want = reference_complex_scan_rope(*(t.to(torch.float32) for t in args))
        assert got.dtype == dtype
        atol = 1e-2 if dtype == torch.float16 else 5e-2
        scale = want.abs().max().clamp(min=1.0).item()
        assert torch.allclose(got.float(), want, atol=atol * scale, rtol=atol)

    def test_deterministic_across_runs(self) -> None:
        args = _rope_inputs(2, 128, 2, 16, 16, 4)
        first = complex_scan_rope(*args)
        for _ in range(4):
            assert torch.equal(complex_scan_rope(*args), first)

    def test_noncontiguous_inputs_match_contiguous(self) -> None:
        args = _rope_inputs(2, 64, 2, 16, 16, 4)
        base = complex_scan_rope(*args)
        x, bb, cc, dt, a, angle = args
        x_nc = x.transpose(1, 2).contiguous().transpose(1, 2)
        b_nc = bb.transpose(2, 3).contiguous().transpose(2, 3)
        assert not (x_nc.is_contiguous() or b_nc.is_contiguous())
        assert torch.equal(complex_scan_rope(x_nc, b_nc, cc, dt, a, angle), base)

    def test_nonfinite_x_masks_match_oracle(self) -> None:
        args = _rope_inputs(2, 64, 2, 8, 16, 4, seed=5)
        x = args[0]
        x[0, 9, 0, 1] = float("inf")
        x[1, 33, 1, 2] = float("nan")
        got = complex_scan_rope(*args)
        want = reference_complex_scan_rope(*args)
        assert torch.equal(torch.isnan(got), torch.isnan(want))
        assert torch.equal(torch.isinf(got), torch.isinf(want))


@pytest.mark.gpu
@requires_gpu
class TestC4NumWarpsCompile:
    def test_compiles_and_matches_at_num_warps_4_and_8(self) -> None:
        # Same #904 framing as C1-C3: no tl.dot anywhere, so the
        # TMEM-promotion pass never engages on sm_100.
        from flash_mamba_rl.kernels.ops import _triton_complex_rope

        args = _rope_inputs(2, 256, 4, 32, 16, 8)
        base = _triton_complex_rope.launch_complex_scan_rope(*args, num_warps=2)
        for warps in (4, 8):
            got = _triton_complex_rope.launch_complex_scan_rope(*args, num_warps=warps)
            assert torch.allclose(got, base, atol=1e-4, rtol=1e-4), warps
        meta = triton_complex_rope_resource_meta()
        assert meta is not None
        assert 0 < meta["n_regs"] <= 255


@pytest.mark.gpu
@requires_gpu
class TestC4ContractGates:
    def test_all_gates_pass_on_cuda(self) -> None:
        args = _rope_inputs(1, 8, 2, 4, 16, 6)
        complex_scan_rope(*args)  # warm the cache
        meta = triton_complex_rope_resource_meta()

        results = verify_rope_op(complex_scan_rope, device="cuda", resource_meta=meta)
        failed = {
            name: (result.reason, result.details)
            for name, result in results.items()
            if not result.passed
        }
        assert not failed, f"rope gates failed on CUDA (resource_meta={meta}): {failed}"
