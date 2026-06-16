"""Triton kernel for the Mamba-3 complex-RoPE selective scan forward (C4).

Fully fused single pass: the official stack splits this op into an
``angle_dt`` cumsum kernel plus a scan that consumes pre-rotated B/C; here
the per-step angle accumulation, tanh/cos/sin, the pairwise rotation and
the decay scan all live in one program — no theta / B_rot / C_rot tensors
ever hit HBM. Same structural rules as C1-C3: no ``tl.dot`` (the sm_100
TMEM-promotion pass never engages), no atomics (one program owns each
output lane; ORD-02 by construction), serial over L.

The rotation is lane-parallel, never a register shuffle: state lane n
belongs to pair k = n // 2; each lane reloads its partner lane (n ^ 1)
from global memory (same cache line) and computes
``self * cos + sign * partner * sin`` with sign -1 on even lanes, +1 on
odd — exactly the reference's interleaved-pair convention. Lanes at
n >= 2 * num_angles pass through unrotated. theta rides as a [BLOCK_N]
fp32 accumulator (both lanes of a pair duplicate it) with a floor-based
remainder 2*pi applied per step: algebraically identical to the
reference's mod-after-cumsum, and better conditioned at large L because
cos/sin always see a small argument while the reference's fp32 cumsum
carries O(L * dt) magnitudes before its single mod.

libdevice transcendentals throughout (exp/tanh/cos/sin), not the
approx-path tl variants, for the same denormal/precision reasons as C1.

Import only with triton installed and CUDA the target — the dispatcher in
``complex_scan_rope.py`` guards both. Layout (enforced via
``.contiguous()``):

    x, y        : [B, L, H, P]   row-major
    B, C        : [B, L, H, N]
    dt          : [B, L, H]
    A           : [H]
    angle_proj  : [B, L, H, S]   (2*S <= N)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from flash_mamba_rl.kernels.autotune import KernelConfig
from flash_mamba_rl.kernels.ops._resource_meta import collect_resource_meta

try:  # moved between minor versions
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover - depends on installed triton
    from triton.language.extra.cuda import libdevice

_TWO_PI = tl.constexpr(6.283185307179586)
_PI = tl.constexpr(3.141592653589793)

# One CTA holds the whole state dim; N above this needs a multi-block design.
MAX_BLOCK_N = 128


@triton.jit  # type: ignore[untyped-decorator]
def _complex_rope_kernel(  # type: ignore[no-untyped-def]
    x_ptr,
    b_ptr,
    c_ptr,
    dt_ptr,
    a_ptr,
    ang_ptr,
    y_ptr,
    seq_len,
    nheads,
    headdim,
    n_state,
    s_angles,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_p = tl.program_id(2)

    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_n = tl.arange(0, BLOCK_N)
    mask_p = offs_p < headdim
    mask_n = offs_n < n_state
    mask_pn = mask_p[:, None] & mask_n[None, :]
    # Lane n belongs to rotation pair k = n // 2; lanes past 2*S identity.
    pair_k = offs_n // 2
    mask_rot = pair_k < s_angles
    partner = offs_n ^ 1
    mask_partner = partner < n_state
    sign = tl.where(offs_n % 2 == 0, -1.0, 1.0)

    a_h = tl.load(a_ptr + pid_h).to(tl.float32)

    h = tl.zeros((BLOCK_P, BLOCK_N), dtype=tl.float32)
    theta = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Running offsets are int64 via the promoted base; the per-step
    # increments are H-scale strides (nheads * max(P, N, S)), int32-safe
    # at any contract-legal shape.
    xhp_off = (pid_b.to(tl.int64) * seq_len * nheads + pid_h) * headdim + offs_p
    bn_off = (pid_b.to(tl.int64) * seq_len * nheads + pid_h) * n_state + offs_n
    bn_partner_off = (pid_b.to(tl.int64) * seq_len * nheads + pid_h) * n_state + partner
    dt_off = pid_b.to(tl.int64) * seq_len * nheads + pid_h
    ang_off = (pid_b.to(tl.int64) * seq_len * nheads + pid_h) * s_angles + pair_k

    for _t in range(seq_len):
        dt_t = tl.load(dt_ptr + dt_off).to(tl.float32)
        ang_t = tl.load(ang_ptr + ang_off, mask=mask_rot, other=0.0).to(tl.float32)

        theta += tl.where(mask_rot, libdevice.tanh(ang_t) * dt_t * _PI, 0.0)
        theta = theta - libdevice.floor(theta / _TWO_PI) * _TWO_PI
        cos_t = libdevice.cos(theta)
        sin_t = libdevice.sin(theta)

        b_self = tl.load(b_ptr + bn_off, mask=mask_n, other=0.0).to(tl.float32)
        b_part = tl.load(b_ptr + bn_partner_off, mask=mask_partner, other=0.0).to(tl.float32)
        c_self = tl.load(c_ptr + bn_off, mask=mask_n, other=0.0).to(tl.float32)
        c_part = tl.load(c_ptr + bn_partner_off, mask=mask_partner, other=0.0).to(tl.float32)
        b_rot = tl.where(mask_rot, b_self * cos_t + sign * b_part * sin_t, b_self)
        c_rot = tl.where(mask_rot, c_self * cos_t + sign * c_part * sin_t, c_self)

        x_t = tl.load(x_ptr + xhp_off, mask=mask_p, other=0.0).to(tl.float32)
        alpha_t = libdevice.exp(dt_t * a_h)

        # Reference grouping ((dt * B_rot) * x) and padded-lane containment
        # as in C1: a non-finite x lane must poison exactly its own row.
        bu = (dt_t * b_rot)[None, :] * x_t[:, None]
        h = tl.where(mask_pn, alpha_t * h + bu, 0.0)

        y_t = tl.sum(h * c_rot[None, :], axis=1)
        tl.store(y_ptr + xhp_off, y_t.to(y_ptr.dtype.element_ty), mask=mask_p)

        xhp_off += nheads * headdim
        bn_off += nheads * n_state
        bn_partner_off += nheads * n_state
        dt_off += nheads
        ang_off += nheads * s_angles


def launch_complex_scan_rope(
    x: Tensor,
    b_proj: Tensor,
    c_proj: Tensor,
    dt: Tensor,
    a: Tensor,
    angle_proj: Tensor,
    *,
    num_warps: int | None = None,
    config: KernelConfig | None = None,
) -> Tensor:
    """Launch the fused rotary scan on CUDA tensors.

    Dispatch keys on ``x`` (device, dtype); every load upcasts to fp32 and
    ``y`` rounds once at store into ``x``'s dtype. ``config`` overrides the
    autotuner's searched knobs (block_p, num_warps, num_stages); ``num_warps``
    is the legacy bench-sweep hook. A None config or None field keeps the
    shipped heuristic.
    """
    batch, seq_len, nheads, headdim = x.shape
    n_state = b_proj.shape[-1]
    s_angles = angle_proj.shape[-1]

    block_n = triton.next_power_of_2(n_state)
    if block_n > MAX_BLOCK_N:
        raise ValueError(f"n_state={n_state} exceeds single-block budget {MAX_BLOCK_N}")
    if 2 * s_angles > n_state:
        raise ValueError(f"rotary_dim={2 * s_angles} exceeds d_state={n_state}")
    block_p = min(64, triton.next_power_of_2(headdim))
    if config is not None and config.block_p is not None:
        block_p = config.block_p

    x_c = x.contiguous()
    b_c = b_proj.contiguous()
    c_c = c_proj.contiguous()
    dt_c = dt.contiguous()
    a_c = a.contiguous()
    ang_c = angle_proj.contiguous()
    y = torch.empty_like(x_c)

    grid = (batch, nheads, triton.cdiv(headdim, block_p))
    # B200 nw2-vs-nw4 medians (scratch/c4_warp_probe.py): the heuristic wins
    # or ties at B2xL256xH4 (0.211 vs 0.233), B2xL1024xH8 (tie) and
    # B4xL2048xH16 (6.29 vs 6.37); at B8xL2048xH32 nw=2 wins (6.50 vs 7.64)
    # — grid-size dependent, not block-size. Autotune is the v2 lever; the
    # num_warps override below serves measured callers meanwhile.
    warps = num_warps if num_warps is not None else (4 if block_p * block_n >= 512 else 2)
    if config is not None and config.num_warps is not None:
        warps = config.num_warps
    extra: dict[str, int] = {}
    if config is not None and config.num_stages is not None:
        extra["num_stages"] = config.num_stages
    _complex_rope_kernel[grid](
        x_c,
        b_c,
        c_c,
        dt_c,
        a_c,
        ang_c,
        y,
        seq_len,
        nheads,
        headdim,
        n_state,
        s_angles,
        BLOCK_P=block_p,
        BLOCK_N=block_n,
        num_warps=warps,
        **extra,
    )
    return y


def resource_meta() -> dict[str, int] | None:
    """Resource envelope across all compiled specialisations of the kernel."""
    return collect_resource_meta(_complex_rope_kernel)
