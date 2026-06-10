"""Triton kernel for the SISO selective-scan forward (C1).

Import this module only when ``triton`` is installed and a CUDA device is
the target — the public dispatcher in ``forward_chunked_scan.py`` guards
both. Layout assumptions (enforced by the launcher via ``.contiguous()``):

    u, delta, y : [B, L, D]   row-major
    A           : [D, N]
    B, C        : [B, L, N]
    D_skip      : [D]

Parallelisation: one program per (batch, D-block). The program carries the
hidden state ``h`` as a [BLOCK_D, BLOCK_N] fp32 register block and walks the
sequence serially — the recurrence over t matches the reference oracle's
accumulation order. All arithmetic is fp32 regardless of input dtype
(fp16/bf16 inputs are upcast at load, output rounds once at store), which is
the contract PRC-02 measures.

Softplus matches ``torch.nn.functional.softplus`` exactly: linear above
threshold 20, ``log1p(exp(x))`` below (libdevice log1p, not log(1+x), so
tiny ``exp(x)`` is not flushed to zero).
"""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl
from torch import Tensor

try:  # moved between minor versions
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover - depends on installed triton
    from triton.language.extra.cuda import libdevice

# torch.nn.functional.softplus switches to the identity above this threshold.
# Instantiated as tl.constexpr: triton >= 3.7 rejects plain globals in jit code.
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

    # int64 batch bases: b * L * D overflows int32 for large batched runs.
    uld_base = pid_b.to(tl.int64) * seq_len * d_model
    bln_base = pid_b.to(tl.int64) * seq_len * n_state

    for t in range(seq_len):
        uld_off = uld_base + t * d_model + offs_d
        u_t = tl.load(u_ptr + uld_off, mask=mask_d, other=0.0).to(tl.float32)
        dlt = tl.load(delta_ptr + uld_off, mask=mask_d, other=0.0).to(tl.float32)

        bln_off = bln_base + t * n_state + offs_n
        b_t = tl.load(b_ptr + bln_off, mask=mask_n, other=0.0).to(tl.float32)
        c_t = tl.load(c_ptr + bln_off, mask=mask_n, other=0.0).to(tl.float32)

        dbar = tl.where(dlt > _SOFTPLUS_THRESHOLD, dlt, libdevice.log1p(tl.exp(dlt)))

        a_bar = tl.exp(dbar[:, None] * a)
        bu = (dbar * u_t)[:, None] * b_t[None, :]
        # Padding lanes must stay exactly zero: with non-finite u, Inf * 0
        # in a padded lane would mint a NaN that poisons the N-reduction.
        bu = tl.where(mask_dn, bu, 0.0)
        h = a_bar * h + bu

        y_t = tl.sum(h * c_t[None, :], axis=1) + d_skip * u_t
        tl.store(y_ptr + uld_off, y_t.to(y_ptr.dtype.element_ty), mask=mask_d)


def launch_forward_scan(
    u: Tensor, delta: Tensor, a: Tensor, b: Tensor, c: Tensor, d_skip: Tensor
) -> Tensor:
    """Launch the Triton scan. Inputs must be CUDA tensors of one dtype."""
    batch, seq_len, d_model = u.shape
    n_state = a.shape[1]

    block_n = triton.next_power_of_2(n_state)
    if block_n > MAX_BLOCK_N:
        raise ValueError(f"n_state={n_state} exceeds single-block budget {MAX_BLOCK_N}")
    block_d = min(64, triton.next_power_of_2(d_model))

    u_c = u.contiguous()
    delta_c = delta.contiguous()
    a_c = a.contiguous()
    b_c = b.contiguous()
    c_c = c.contiguous()
    d_c = d_skip.contiguous()
    y = torch.empty_like(u_c)

    grid = (batch, triton.cdiv(d_model, block_d))
    num_warps = 4 if block_d * block_n >= 512 else 2
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
    )
    return y


def resource_meta() -> dict[str, int] | None:
    """Best-effort resource metadata from the compiled kernel cache.

    Reads ``n_regs`` / ``n_spills`` / shared-memory bytes off the most
    recently compiled specialisation of the scan kernel, in the shape
    ``gate_res_02_resource_limits`` expects. Returns None when nothing
    has been compiled yet or the (version-dependent) cache layout has
    drifted — absence of evidence must not fabricate evidence.
    """
    caches = getattr(_fwd_scan_kernel, "device_caches", None)
    compiled: list[Any] = []
    if isinstance(caches, dict):
        for entry in caches.values():
            # 3.x: device_caches[device] is a tuple whose first slot is the
            # signature -> CompiledKernel dict.
            cache_dict = entry[0] if isinstance(entry, tuple) else entry
            if isinstance(cache_dict, dict):
                compiled.extend(cache_dict.values())
    legacy = getattr(_fwd_scan_kernel, "cache", None)
    if isinstance(legacy, dict):
        for cache_dict in legacy.values():
            if isinstance(cache_dict, dict):
                compiled.extend(cache_dict.values())

    for kernel in reversed(compiled):
        n_regs = getattr(kernel, "n_regs", None)
        if n_regs is None:
            continue
        meta: dict[str, int] = {"n_regs": int(n_regs)}
        n_spills = getattr(kernel, "n_spills", None)
        if n_spills is not None:
            # ptxas reports spills in bytes; triton surfaces the raw figure.
            meta["spill_bytes"] = int(n_spills)
        shared = getattr(getattr(kernel, "metadata", None), "shared", None)
        if shared is not None:
            meta["shared_bytes"] = int(shared)
        return meta
    return None
