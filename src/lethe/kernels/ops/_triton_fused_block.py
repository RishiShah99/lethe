"""Triton kernels for the Mamba fused-block forward (C5)."""

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
# The x window materialises as a [BLOCK_D, BLOCK_K] register block; suite never exceeds K=4.
MAX_CONV_K = 8
# Kernel B's load tile is [BLOCK_T, BLOCK_D_NORM] fp32; cap the product to ~8 KB/program.
_NORM_TILE = 2048
_MAX_BLOCK_D_NORM = 256


@triton.jit  # type: ignore[untyped-decorator]
def _conv_scan_kernel(  # type: ignore[no-untyped-def]
    x_ptr,
    conv_w_ptr,
    conv_b_ptr,
    delta_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    dskip_ptr,
    y_scan_ptr,
    l_out,
    d_model,
    n_state,
    CONV_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_d = offs_d < d_model
    mask_n = offs_n < n_state
    mask_k = offs_k < CONV_K
    mask_dn = mask_d[:, None] & mask_n[None, :]
    mask_dk = mask_d[:, None] & mask_k[None, :]

    a = tl.load(a_ptr + offs_d[:, None] * n_state + offs_n[None, :], mask=mask_dn, other=0.0).to(
        tl.float32
    )
    d_skip = tl.load(dskip_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    conv_w = tl.load(
        conv_w_ptr + offs_d[:, None] * CONV_K + offs_k[None, :], mask=mask_dk, other=0.0
    ).to(tl.float32)
    conv_b = tl.load(conv_b_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)

    h = tl.zeros((BLOCK_D, BLOCK_N), dtype=tl.float32)

    seq_in = l_out + CONV_K - 1
    x_off = pid_b.to(tl.int64) * seq_in * d_model + offs_d
    od_off = pid_b.to(tl.int64) * l_out * d_model + offs_d
    bln_off = pid_b.to(tl.int64) * l_out * n_state + offs_n

    for _t in range(l_out):
        xw = tl.load(
            x_ptr + x_off[:, None] + offs_k[None, :] * d_model, mask=mask_dk, other=0.0
        ).to(tl.float32)
        conv = tl.sum(conv_w * xw, axis=1) + conv_b
        # Full-precision exp (not ex2.approx) matches torch's non-finite semantics: silu(-Inf)=NaN.
        z = conv * (1.0 / (1.0 + libdevice.exp(-conv)))

        dlt = tl.load(delta_ptr + od_off, mask=mask_d, other=0.0).to(tl.float32)
        b_t = tl.load(b_ptr + bln_off, mask=mask_n, other=0.0).to(tl.float32)
        c_t = tl.load(c_ptr + bln_off, mask=mask_n, other=0.0).to(tl.float32)

        dbar = tl.where(dlt > _SOFTPLUS_THRESHOLD, dlt, libdevice.log1p(libdevice.exp(dlt)))

        a_bar = libdevice.exp(dbar[:, None] * a)
        bu = (dbar * z)[:, None] * b_t[None, :]
        # Padding lanes must stay exactly zero; a non-finite z there would poison the N-reduction.
        h = tl.where(mask_dn, a_bar * h + bu, 0.0)

        y_t = tl.sum(h * c_t[None, :], axis=1) + d_skip * z
        tl.store(y_scan_ptr + od_off, y_t, mask=mask_d)

        x_off += d_model
        od_off += d_model
        bln_off += n_state


@triton.jit  # type: ignore[untyped-decorator]
def _rmsnorm_kernel(  # type: ignore[no-untyped-def]
    y_scan_ptr,
    w_ptr,
    out_ptr,
    l_out,
    d_model,
    eps,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < l_out
    row_off = (pid_b.to(tl.int64) * l_out + offs_t) * d_model

    ssq = tl.zeros((BLOCK_T,), dtype=tl.float32)
    for d0 in range(0, d_model, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask = mask_t[:, None] & (offs_d < d_model)[None, :]
        v = tl.load(y_scan_ptr + row_off[:, None] + offs_d[None, :], mask=mask, other=0.0)
        ssq += tl.sum(v * v, axis=1)

    rms = libdevice.sqrt(ssq / d_model + eps)

    for d0 in range(0, d_model, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask_d = offs_d < d_model
        mask = mask_t[:, None] & mask_d[None, :]
        v = tl.load(y_scan_ptr + row_off[:, None] + offs_d[None, :], mask=mask, other=0.0)
        w = tl.load(w_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        out = v / rms[:, None] * w[None, :]
        tl.store(
            out_ptr + row_off[:, None] + offs_d[None, :],
            out.to(out_ptr.dtype.element_ty),
            mask=mask,
        )


def launch_fused_block_forward(
    x: Tensor,
    conv_weight: Tensor,
    conv_bias: Tensor,
    delta: Tensor,
    a: Tensor,
    b: Tensor,
    c: Tensor,
    d_skip: Tensor,
    norm_weight: Tensor,
    eps: float,
    num_warps: int | None = None,
    *,
    config: KernelConfig | None = None,
) -> Tensor:
    """Launch the fused block."""
    batch, seq_in, d_model = x.shape
    conv_k = conv_weight.shape[-1]
    n_state = a.shape[1]
    l_out = seq_in - (conv_k - 1)

    if conv_k > MAX_CONV_K:
        raise ValueError(f"conv kernel size {conv_k} exceeds window budget {MAX_CONV_K}")
    if l_out < 1:
        raise ValueError(f"sequence length {seq_in} shorter than conv window {conv_k}")
    block_n = triton.next_power_of_2(n_state)
    if block_n > MAX_BLOCK_N:
        raise ValueError(f"n_state={n_state} exceeds single-block budget {MAX_BLOCK_N}")
    # block_d only tiles D into independent per-D outputs; it never changes the math.
    block_d = min(16, triton.next_power_of_2(d_model))
    if config is not None and config.block_d is not None:
        block_d = config.block_d
    block_k = triton.next_power_of_2(conv_k)

    x_c = x.contiguous()
    conv_w_c = conv_weight.contiguous()
    conv_b_c = conv_bias.contiguous()
    delta_c = delta.contiguous()
    a_c = a.contiguous()
    b_c = b.contiguous()
    c_c = c.contiguous()
    dskip_c = d_skip.contiguous()
    norm_w_c = norm_weight.contiguous()

    y_scan = torch.empty(batch, l_out, d_model, device=x.device, dtype=torch.float32)
    out = torch.empty(batch, l_out, d_model, device=x.device, dtype=x.dtype)

    grid_scan = (batch, triton.cdiv(d_model, block_d))
    # num_warps=4 wins universally here; block_d=16 alone measured 1.25x, with nw=4 together 1.64x.
    num_warps_scan = num_warps if num_warps is not None else 4
    if config is not None and config.num_warps is not None:
        num_warps_scan = config.num_warps
    extra: dict[str, int] = {}
    if config is not None and config.num_stages is not None:
        extra["num_stages"] = config.num_stages
    _conv_scan_kernel[grid_scan](
        x_c,
        conv_w_c,
        conv_b_c,
        delta_c,
        a_c,
        b_c,
        c_c,
        dskip_c,
        y_scan,
        l_out,
        d_model,
        n_state,
        CONV_K=conv_k,
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        BLOCK_N=block_n,
        num_warps=num_warps_scan,
        **extra,
    )

    block_d_norm = min(_MAX_BLOCK_D_NORM, triton.next_power_of_2(d_model))
    block_t = max(1, _NORM_TILE // block_d_norm)
    grid_norm = (batch, triton.cdiv(l_out, block_t))
    num_warps_norm = 4 if block_t * block_d_norm >= 512 else 2
    _rmsnorm_kernel[grid_norm](
        y_scan,
        norm_w_c,
        out,
        l_out,
        d_model,
        eps,
        BLOCK_T=block_t,
        BLOCK_D=block_d_norm,
        num_warps=num_warps_norm,
    )
    return out


def resource_meta() -> dict[str, int] | None:
    """Max-envelope resource metadata over both kernels' compiled specialisations."""
    merged: dict[str, int] | None = None
    for jit_fn in (_conv_scan_kernel, _rmsnorm_kernel):
        meta = collect_resource_meta(jit_fn)
        if meta is None:
            continue
        if merged is None:
            merged = dict(meta)
        else:
            for key, val in meta.items():
                merged[key] = max(merged.get(key, 0), val)
    return merged
