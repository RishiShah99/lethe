"""Chunk-parallel-carry SISO forward scan (the long-L speedup lever)."""

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

_SOFTPLUS_THRESHOLD = tl.constexpr(20.0)
MAX_BLOCK_N = 128


@triton.jit  # type: ignore[untyped-decorator]
def _chunk_reduce_kernel(  # type: ignore[no-untyped-def]
    u_ptr,
    delta_ptr,
    a_ptr,
    b_ptr,
    sloc_ptr,
    adecay_ptr,
    seq_len,
    d_model,
    n_state,
    n_chunks,
    chunk_len,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    mask_d = offs_d < d_model
    mask_n = offs_n < n_state
    mask_dn = mask_d[:, None] & mask_n[None, :]

    a = tl.load(a_ptr + offs_d[:, None] * n_state + offs_n[None, :], mask=mask_dn, other=0.0).to(
        tl.float32
    )

    h = tl.zeros((BLOCK_D, BLOCK_N), dtype=tl.float32)
    decay = tl.where(mask_dn, 1.0, 0.0)

    t0 = pid_c * chunk_len
    uld_off = pid_b.to(tl.int64) * seq_len * d_model + t0.to(tl.int64) * d_model + offs_d
    bln_off = pid_b.to(tl.int64) * seq_len * n_state + t0.to(tl.int64) * n_state + offs_n

    for _j in range(chunk_len):
        u_t = tl.load(u_ptr + uld_off, mask=mask_d, other=0.0).to(tl.float32)
        dlt = tl.load(delta_ptr + uld_off, mask=mask_d, other=0.0).to(tl.float32)
        b_t = tl.load(b_ptr + bln_off, mask=mask_n, other=0.0).to(tl.float32)

        dbar = tl.where(dlt > _SOFTPLUS_THRESHOLD, dlt, libdevice.log1p(libdevice.exp(dlt)))
        a_bar = libdevice.exp(dbar[:, None] * a)
        bu = (dbar * u_t)[:, None] * b_t[None, :]
        h = tl.where(mask_dn, a_bar * h + bu, 0.0)
        decay = tl.where(mask_dn, decay * a_bar, 0.0)

        uld_off += d_model
        bln_off += n_state

    state_off = ((pid_b.to(tl.int64) * n_chunks + pid_c) * d_model + offs_d[:, None]) * n_state
    state_off = state_off + offs_n[None, :]
    tl.store(sloc_ptr + state_off, h, mask=mask_dn)
    tl.store(adecay_ptr + state_off, decay, mask=mask_dn)


@triton.jit  # type: ignore[untyped-decorator]
def _chunk_scan_kernel(  # type: ignore[no-untyped-def]
    u_ptr,
    delta_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    dskip_ptr,
    hin_ptr,
    y_ptr,
    seq_len,
    d_model,
    n_state,
    n_chunks,
    chunk_len,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    mask_d = offs_d < d_model
    mask_n = offs_n < n_state
    mask_dn = mask_d[:, None] & mask_n[None, :]

    a = tl.load(a_ptr + offs_d[:, None] * n_state + offs_n[None, :], mask=mask_dn, other=0.0).to(
        tl.float32
    )
    d_skip = tl.load(dskip_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)

    state_off = ((pid_b.to(tl.int64) * n_chunks + pid_c) * d_model + offs_d[:, None]) * n_state
    state_off = state_off + offs_n[None, :]
    h = tl.load(hin_ptr + state_off, mask=mask_dn, other=0.0).to(tl.float32)
    h = tl.where(mask_dn, h, 0.0)

    t0 = pid_c * chunk_len
    uld_off = pid_b.to(tl.int64) * seq_len * d_model + t0.to(tl.int64) * d_model + offs_d
    bln_off = pid_b.to(tl.int64) * seq_len * n_state + t0.to(tl.int64) * n_state + offs_n

    for _j in range(chunk_len):
        u_t = tl.load(u_ptr + uld_off, mask=mask_d, other=0.0).to(tl.float32)
        dlt = tl.load(delta_ptr + uld_off, mask=mask_d, other=0.0).to(tl.float32)
        b_t = tl.load(b_ptr + bln_off, mask=mask_n, other=0.0).to(tl.float32)
        c_t = tl.load(c_ptr + bln_off, mask=mask_n, other=0.0).to(tl.float32)

        dbar = tl.where(dlt > _SOFTPLUS_THRESHOLD, dlt, libdevice.log1p(libdevice.exp(dlt)))
        a_bar = libdevice.exp(dbar[:, None] * a)
        bu = (dbar * u_t)[:, None] * b_t[None, :]
        h = tl.where(mask_dn, a_bar * h + bu, 0.0)

        y_t = tl.sum(h * c_t[None, :], axis=1) + d_skip * u_t
        tl.store(y_ptr + uld_off, y_t.to(y_ptr.dtype.element_ty), mask=mask_d)

        uld_off += d_model
        bln_off += n_state


def _carry_scan(sloc: Tensor, adecay: Tensor) -> Tensor:
    """Phase 2: serial carry over the nc chunk boundaries (fp32, deterministic)."""
    batch, n_chunks, d_model, n_state = sloc.shape
    hin = torch.empty_like(sloc)
    carry = torch.zeros(batch, d_model, n_state, dtype=sloc.dtype, device=sloc.device)
    for c in range(n_chunks):
        hin[:, c] = carry
        carry = adecay[:, c] * carry + sloc[:, c]
    return hin


def launch_chunk_parallel_scan(
    u: Tensor,
    delta: Tensor,
    a: Tensor,
    b: Tensor,
    c: Tensor,
    d_skip: Tensor,
    *,
    chunk_len: int,
    config: KernelConfig | None = None,
) -> Tensor:
    """Chunk-parallel forward scan."""
    batch, seq_len, d_model = u.shape
    n_state = a.shape[1]
    if seq_len % chunk_len != 0:
        raise ValueError(f"seq_len {seq_len} must be divisible by chunk_len {chunk_len}")
    n_chunks = seq_len // chunk_len

    block_n = triton.next_power_of_2(n_state)
    if block_n > MAX_BLOCK_N:
        raise ValueError(f"n_state={n_state} exceeds single-block budget {MAX_BLOCK_N}")
    block_d = min(64, triton.next_power_of_2(d_model))
    if config is not None and config.block_d is not None:
        block_d = config.block_d

    u_c = u.contiguous()
    delta_c = delta.contiguous()
    a_c = a.contiguous()
    b_c = b.contiguous()
    c_c = c.contiguous()
    d_c = d_skip.contiguous()
    y = torch.empty_like(u_c)

    sloc = torch.empty(batch, n_chunks, d_model, n_state, dtype=torch.float32, device=u.device)
    adecay = torch.empty_like(sloc)

    grid = (batch, n_chunks, triton.cdiv(d_model, block_d))
    num_warps = 4 if block_d * block_n >= 512 else 2
    if config is not None and config.num_warps is not None:
        num_warps = config.num_warps
    extra: dict[str, int] = {}
    if config is not None and config.num_stages is not None:
        extra["num_stages"] = config.num_stages

    _chunk_reduce_kernel[grid](
        u_c, delta_c, a_c, b_c, sloc, adecay,
        seq_len, d_model, n_state, n_chunks, chunk_len,
        BLOCK_D=block_d, BLOCK_N=block_n, num_warps=num_warps, **extra,
    )  # fmt: skip

    hin = _carry_scan(sloc, adecay)

    _chunk_scan_kernel[grid](
        u_c, delta_c, a_c, b_c, c_c, d_c, hin, y,
        seq_len, d_model, n_state, n_chunks, chunk_len,
        BLOCK_D=block_d, BLOCK_N=block_n, num_warps=num_warps, **extra,
    )  # fmt: skip
    return y


def resource_meta() -> dict[str, int] | None:
    """Resource envelope over both compiled chunk-parallel kernels."""
    rm: dict[str, int] = {}
    for kern in (_chunk_reduce_kernel, _chunk_scan_kernel):
        meta = collect_resource_meta(kern)
        if meta is None:
            continue
        for key, val in meta.items():
            rm[key] = max(rm.get(key, 0), val)
    return rm or None
