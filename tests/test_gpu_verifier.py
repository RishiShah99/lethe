"""GPU-side validation of the verifier."""

import sys

import pytest
import torch

from lethe.verifier.compile import ErrorClass, compile_kernel
from lethe.verifier.sandbox import run_in_subprocess
from lethe.verifier.timing import benchmark

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="RLIMIT_AS is POSIX-only")


VALID_KERNEL_WITH_WARMUP = """
import torch
import triton
import triton.language as tl


@triton.jit
def add_one_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + 1, mask=mask)


def __warmup__():
    x = torch.arange(64, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    add_one_kernel[(1,)](x, out, 64, BLOCK=64)
    torch.cuda.synchronize()
    assert torch.equal(out, x + 1)
"""

# tl.arange bounds must be powers of 2; BLOCK=7 fails at JIT compile time, inside __warmup__.
BROKEN_AT_LAUNCH = """
import torch
import triton
import triton.language as tl


@triton.jit
def bad_kernel(x_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(x_ptr + offs, 0.0)


def __warmup__():
    x = torch.zeros(8, device="cuda")
    bad_kernel[(1,)](x, BLOCK=7)
"""

CANDIDATE_WITH_MISSING_IMPORT = "import does_not_exist_xyz_fmrl\n"


@pytest.mark.gpu
@requires_gpu
class TestCompileOnGPU:
    def test_valid_kernel_with_warmup_compiles(self) -> None:
        result = compile_kernel(VALID_KERNEL_WITH_WARMUP, timeout_s=120.0)
        assert result.success, result.stderr
        assert result.error_class is ErrorClass.OK
        assert not result.ptxas_c7907

    def test_launch_time_compile_failure_is_caught(self) -> None:
        result = compile_kernel(BROKEN_AT_LAUNCH, timeout_s=120.0)
        assert not result.success
        assert result.error_class is not ErrorClass.OK
        # stderr must carry the actual launch-time diagnostic; empty stderr means no evidence.
        assert result.stderr.strip(), "launch failure produced no stderr evidence"

    def test_candidate_import_error_does_not_pass(self) -> None:
        # candidate ImportError must fail compile, not fall through to the CPU ast.parse path.
        result = compile_kernel(CANDIDATE_WITH_MISSING_IMPORT, timeout_s=120.0)
        assert not result.success


@pytest.mark.gpu
@requires_gpu
class TestTimingOnGPU:
    def test_cuda_event_benchmark(self) -> None:
        a = torch.randn(1024, 1024, device="cuda")
        b = torch.randn(1024, 1024, device="cuda")
        result = benchmark(torch.mm, (a, b), warmup=5, trials=20)
        assert result.n_trials == 20
        assert 0.0 < result.median_ms < 1000.0
        assert result.min_ms <= result.median_ms <= result.max_ms

    def test_larger_work_takes_longer(self) -> None:
        small = torch.randn(256, 256, device="cuda")
        large = torch.randn(4096, 4096, device="cuda")
        t_small = benchmark(torch.mm, (small, small), warmup=5, trials=20)
        t_large = benchmark(torch.mm, (large, large), warmup=5, trials=20)
        assert t_large.median_ms > t_small.median_ms


@pytest.mark.gpu
@requires_gpu
class TestSandboxOnGPU:
    def test_runs_gpu_callable(self) -> None:
        x = torch.arange(16, dtype=torch.float32)
        # memory_limit_mb=0: CUDA context init maps too much VA space for RLIMIT_AS to survive.
        result = run_in_subprocess(
            "tests._gpu_helpers", "gpu_square", (x,), timeout_s=300.0, memory_limit_mb=0
        )
        assert result.success, result.stderr
        assert torch.equal(result.output, x**2)


@posix_only
def test_sandbox_posix_memory_limit_enforced() -> None:
    result = run_in_subprocess(
        "tests._sandbox_helpers",
        "alloc_8_gib",
        (),
        timeout_s=60.0,
        memory_limit_mb=1024,
    )
    assert not result.success
    assert result.error_class is ErrorClass.OOM
