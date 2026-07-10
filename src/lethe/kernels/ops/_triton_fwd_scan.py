"""Triton kernel for the SISO selective-scan forward (C1)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from lethe.kernels.autotune import KernelConfig
from lethe.kernels.ops._resource_meta import collect_resource_meta

try:  # moved between minor versions
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover - depends on installed triton
    from triton.language.extra.cuda import libdevice

# torch.nn.functional.softplus switches to the identity above this threshold.
_SOFTPLUS_THRESHOLD = tl.constexpr(20.0)

# One CTA holds the whole state dim; N above this needs a multi-block design.
MAX_BLOCK_N = 128


@triton.jit  # type: ignore[untyped-decorator]
def _fwd_scan_kernel(  # type: ignore[no-untyped-def]
    u_ptr,
    delta_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    dskip_ptr,
    y_ptr,
    seq_len,
    d_model,
    n_state,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    mask_d = offs_d < d_model
    mask_n = offs_n < n_state
    mask_dn = mask_d[:, None] & mask_n[None, :]

    # Time-invariant operands, loaded once.
    a = tl.load(a_ptr + offs_d[:, None] * n_state + offs_n[None, :], mask=mask_dn, other=0.0).to(
        tl.float32
    )
    d_skip = tl.load(dskip_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)

    h = tl.zeros((BLOCK_D, BLOCK_N), dtype=tl.float32)

    # int64 offsets avoid overflow: b * L * D exceeds int32 range for large batched runs.
    uld_off = pid_b.to(tl.int64) * seq_len * d_model + offs_d
    bln_off = pid_b.to(tl.int64) * seq_len * n_state + offs_n

    for _t in range(seq_len):
        u_t = tl.load(u_ptr + uld_off, mask=mask_d, other=0.0).to(tl.float32)
        dlt = tl.load(delta_ptr + uld_off, mask=mask_d, other=0.0).to(tl.float32)

        b_t = tl.load(b_ptr + bln_off, mask=mask_n, other=0.0).to(tl.float32)
        c_t = tl.load(c_ptr + bln_off, mask=mask_n, other=0.0).to(tl.float32)

        # libdevice.exp, not tl.exp's ex2.approx: the approx flushes subnormals to NaN (EXC-01).
        dbar = tl.where(dlt > _SOFTPLUS_THRESHOLD, dlt, libdevice.log1p(libdevice.exp(dlt)))

        a_bar = libdevice.exp(dbar[:, None] * a)
        bu = (dbar * u_t)[:, None] * b_t[None, :]
        # Padding lanes must stay exactly zero; non-finite u or delta there would mint a poisoning NaN.
        h = tl.where(mask_dn, a_bar * h + bu, 0.0)

        y_t = tl.sum(h * c_t[None, :], axis=1) + d_skip * u_t
        tl.store(y_ptr + uld_off, y_t.to(y_ptr.dtype.element_ty), mask=mask_d)

        uld_off += d_model
        bln_off += n_state


def launch_forward_scan(
    u: Tensor,
    delta: Tensor,
    a: Tensor,
    b: Tensor,
    c: Tensor,
    d_skip: Tensor,
    *,
    config: KernelConfig | None = None,
) -> Tensor:
    """Launch the Triton scan."""
    batch, seq_len, d_model = u.shape
    n_state = a.shape[1]

    block_n = triton.next_power_of_2(n_state)
    if block_n > MAX_BLOCK_N:
        raise ValueError(f"n_state={n_state} exceeds single-block budget {MAX_BLOCK_N}")
    block_d = min(64, triton.next_power_of_2(d_model))
    if config is not None and config.block_d is not None:
        block_d = config.block_d
        # The grid assumes a power-of-two block_d (default next_power_of_2); non-pow2 overrides break it.
        if block_d < 1 or block_d & (block_d - 1):
            raise ValueError(f"block_d override {block_d} must be a positive power of two")

    u_c = u.contiguous()
    delta_c = delta.contiguous()
    a_c = a.contiguous()
    b_c = b.contiguous()
    c_c = c.contiguous()
    d_c = d_skip.contiguous()
    y = torch.empty_like(u_c)

    grid = (batch, triton.cdiv(d_model, block_d))
    num_warps = 4 if block_d * block_n >= 512 else 2
    if config is not None and config.num_warps is not None:
        num_warps = config.num_warps
    extra: dict[str, int] = {}
    if config is not None and config.num_stages is not None:
        extra["num_stages"] = config.num_stages
    _fwd_scan_kernel[grid](
        u_c,
        delta_c,
        a_c,
        b_c,
        c_c,
        d_c,
        y,
        seq_len,
        d_model,
        n_state,
        BLOCK_D=block_d,
        BLOCK_N=block_n,
        num_warps=num_warps,
        **extra,
    )
    return y


def resource_meta() -> dict[str, int] | None:
    """Resource envelope across all compiled specialisations of the kernel."""
    return collect_resource_meta(_fwd_scan_kernel)
