"""Native Blackwell (sm_100) CuTe/tcgen05 GDN-2 training backward — dispatch shim.

Phase 2/3 (PROJECT_PLAN). Two hard kernels carry the novelty:

- the reverse inter-chunk state recurrence — ``dS`` [d_k, d_v] fp32 accumulator
  carried across the reverse scan (the clean whitespace cuLA leaves in Triton);
- the WY / triangular-inverse VJP — the inverse adjoint via two triangular GEMMs,
  ``T`` reused, never re-inverted.

They are wired into the full backward by ``kernels.cute.gdn2_assemble`` (a two-stage
VJP splice over the chunkwise forward; the kernels do the two hard stages, torch the
supporting ones). This shim is the dispatch boundary: on a Blackwell box, for the
**scalar-reducible** regime (``b = w = beta·1``, ``g`` channel-constant) it runs the
assembly with the compiled tcgen05 kernels; for genuinely channel-wise inputs it
returns ``None`` and the caller falls back to the oracle-faithful eager path (the
channel-wise crown is Phase 3).

``is_available`` stays ``False`` until the assembly is green through the reduction
gate on silicon (HANDOFF: lift only once green on box). Off-box, and until lifted,
``native_gdn2_backward`` returns ``None`` — so the verifier grades the eager path and
the kernel slots in transparently once Phase 2 lands. The Phase-2 credential itself
runs the assembly directly (``gdn2_assemble.assembled_scalar_gdn2_backward`` with the
box kernels) through the scalar reduction gate, independent of this flag.
"""

from __future__ import annotations

import importlib

import torch
from torch import Tensor

from flash_mamba_rl.kernels.cute.gdn2_assemble import (
    K1Fn,
    K1FnCW,
    K2Fn,
    K2FnCW,
    assembled_channelwise_gdn2_backward,
    assembled_scalar_gdn2_backward,
)
from flash_mamba_rl.kernels.references.gdn_backward import Gdn2Grads

SUPPORTED_DTYPES: tuple[torch.dtype, ...] = (torch.bfloat16, torch.float16, torch.float32)

# The compiled tcgen05 kernels target these tile dims (gdn2_bwd_{dhu,wy}*.py, this package).
_KERNEL_D_K = 128
_KERNEL_D_V = 64
_KERNEL_CHUNK = 64
# The channel-wise (Phase-3) kernels route GEMMs through the same (128,64,128) config with
# N-tiling, so the value axis may be 64 or 128 (the crown target is d_k=d_v=128).
_KERNEL_D_V_CW: tuple[int, ...] = (64, 128)


def is_available(device: torch.device | None = None) -> bool:
    """True iff the compiled sm_100 assembly is green AND ``device`` is Blackwell.

    The integration gate passed on a B200 (worst scale_rel 3.29e-3 vs the oracle,
    bit-deterministic; results/gdn2_integration_box.json), so the assembly is lifted.
    Gates on sm_100 — compute-capability major 10 (the tcgen05 tier the kernels target;
    excludes consumer Blackwell sm_120, which has no tcgen05). Off-CUDA -> False, so the
    caller keeps the eager fallback and the local (CPU) gates are unaffected.
    """
    if not torch.cuda.is_available():
        return False
    if device is not None and device.type != "cuda":
        return False
    major, _minor = torch.cuda.get_device_capability(device)
    return major == 10


def _is_scalar_reducible(g: Tensor, b: Tensor, w: Tensor) -> bool:
    """True iff ``g`` is channel-constant and ``b == w == beta·1`` (the Phase-2 regime).

    Phase-2 broadcasts a scalar across the channel axis *exactly* (``expand``), so the
    test is bitwise equality — ``allclose`` would misroute a merely *near*-constant
    channel-wise input to the scalar assembly, silently returning the wrong grad_g/b/w.
    """
    return bool(
        torch.equal(g, g[..., :1].expand_as(g))
        and torch.equal(b, b[..., :1].expand_as(b))
        and torch.equal(w, w[..., :1].expand_as(w))
        and torch.equal(b[..., 0], w[..., 0])
    )


def _load_box_kernels() -> tuple[K1Fn, K2Fn]:
    """Lazily load the tcgen05 K#1/K#2 kernels (box-only; importlib keeps src typed).

    The DSL kernel modules live beside this shim but import ``cutlass`` only under a
    guarded ``try`` — so they stay off the package's import graph and are pulled in
    lazily, on a Blackwell box, exactly when dispatched. Returns ``(run_k1_incB, run_k2)``.
    """
    k1: K1Fn = importlib.import_module("flash_mamba_rl.kernels.cute.gdn2_bwd_dhu").run_k1_incB
    k2: K2Fn = importlib.import_module("flash_mamba_rl.kernels.cute.gdn2_bwd_wy").run_k2
    return k1, k2


def _load_box_kernels_cw() -> tuple[K1FnCW, K2FnCW]:
    """Lazily load the channel-wise (Phase-3) tcgen05 K#1/K#2 kernels (box-only)."""
    k1: K1FnCW = importlib.import_module("flash_mamba_rl.kernels.cute.gdn2_bwd_dhu_cw").run_k1_incB
    k2: K2FnCW = importlib.import_module("flash_mamba_rl.kernels.cute.gdn2_bwd_wy_cw").run_k2
    return k1, k2


def native_gdn2_backward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    b: Tensor,
    w: Tensor,
    do: Tensor,
    *,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
) -> Gdn2Grads | None:
    """Six GDN-2 gradients from the native CuTe assembly, or ``None`` if unavailable.

    Signature mirrors ``reference_gdn2_backward``. Returns ``None`` (the fallback
    contract) when the kernel is absent, the device is not Blackwell, the dtype is
    unsupported, or the tile dims do not match the kernels. The scalar-reducible regime
    (``b = w = beta``, ``g`` channel-constant) routes through the Phase-2 scalar assembly
    (d_v=64); the genuinely channel-wise regime routes through the Phase-3 crown assembly
    (d_v in {64, 128}). Shapes: ``q``/``k``/``g``/``b`` [B, L, H, d_k]; ``v``/``w``/``do``
    [B, L, H, d_v].
    """
    if not is_available(q.device) or q.dtype not in SUPPORTED_DTYPES:
        return None
    if g.shape[-1] != _KERNEL_D_K or q.shape[1] % _KERNEL_CHUNK != 0:
        return None

    if _is_scalar_reducible(g, b, w):
        if w.shape[-1] != _KERNEL_D_V:  # the scalar tcgen05 kernels are dim-locked to d_v=64
            return None
        k1_fn, k2_fn = _load_box_kernels()
        return assembled_scalar_gdn2_backward(
            q,
            k,
            v,
            g,
            b,
            w,
            do,
            scale=scale,
            use_qk_l2norm=use_qk_l2norm,
            k1_fn=k1_fn,
            k2_fn=k2_fn,
        )

    if w.shape[-1] not in _KERNEL_D_V_CW:
        return None
    k1_cw, k2_cw = _load_box_kernels_cw()
    return assembled_channelwise_gdn2_backward(
        q,
        k,
        v,
        g,
        b,
        w,
        do,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm,
        k1_fn=k1_cw,
        k2_fn=k2_cw,
    )
