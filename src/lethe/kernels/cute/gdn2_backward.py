"""Native Blackwell (sm_100) CuTe/tcgen05 GDN-2 training backward, dispatch shim.

Two kernels do the hard work: the reverse inter-chunk state recurrence (``dS``
[d_k, d_v] fp32 accumulator carried across the reverse scan; cuLA imports this from
fla in Triton, no native sm_100 implementation exists there) and the WY /
triangular-inverse VJP (the inverse adjoint via two triangular GEMMs, ``T`` reused,
never re-inverted).

They are wired into the full backward by ``kernels.cute.gdn2_assemble`` (a two-stage
VJP splice over the chunkwise forward; the kernels do the two hard stages, torch the
supporting ones). This shim is the dispatch boundary: on Blackwell hardware, for the
**scalar-reducible** regime (``b = w = beta·1``, ``g`` channel-constant) it runs the
assembly with the compiled tcgen05 kernels; for genuinely channel-wise inputs it
returns ``None`` and the caller falls back to the oracle-faithful eager path (the
channel-wise regime is Phase 3).

``is_available`` stays ``False`` until the assembly passes the reduction gate on
hardware. Until then ``native_gdn2_backward`` returns ``None``, so the verifier
grades the eager path and the kernel slots in transparently once it lands. The
Phase-2 credential runs the assembly directly
(``gdn2_assemble.assembled_scalar_gdn2_backward`` with the compiled kernels) through
the scalar reduction gate, independent of this flag.
"""

from __future__ import annotations

import importlib

import torch
from torch import Tensor

from lethe.kernels.cute.gdn2_assemble import (
    K1Fn,
    K1FnCW,
    K2Fn,
    K2FnCW,
    assembled_channelwise_gdn2_backward,
    assembled_scalar_gdn2_backward,
)
from lethe.kernels.cute.gdn2_family import (
    gla_backward,
    kda_backward,
    la_backward,
    ssd_backward,
)
from lethe.kernels.references.family_oracles import (
    GlaGrads,
    KdaGrads,
    LaGrads,
    SsdGrads,
)
from lethe.kernels.references.gdn2_chunkwise_cw import ChunkwiseForwardCW
from lethe.kernels.references.gdn_backward import Gdn2Grads

SUPPORTED_DTYPES: tuple[torch.dtype, ...] = (torch.bfloat16, torch.float16, torch.float32)

# The compiled tcgen05 kernels target these tile dims (gdn2_bwd_{dhu,wy}*.py, this package).
_KERNEL_D_K = 128
_KERNEL_D_V = 64
_KERNEL_CHUNK = 64
# The channel-wise (Phase-3) kernels route GEMMs through the same (128,64,128) config with
# N-tiling, so the value axis may be 64 or 128 (the target shape is d_k=d_v=128).
_KERNEL_D_V_CW: tuple[int, ...] = (64, 128)


def is_available(device: torch.device | None = None) -> bool:
    """True iff the compiled sm_100 assembly is validated AND ``device`` is Blackwell.

    The integration gate passed on a B200 (worst scale_rel 3.29e-3 vs the oracle,
    bit-deterministic), so the assembly is enabled. Gates on sm_100, compute-capability
    major 10 (the tcgen05 tier the kernels target; excludes consumer Blackwell sm_120,
    which has no tcgen05). Off-CUDA -> False, so the caller keeps the eager fallback and
    the local (CPU) gates are unaffected.
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
    test is bitwise equality; ``allclose`` would misroute a merely *near*-constant
    channel-wise input to the scalar assembly, silently returning the wrong grad_g/b/w.
    """
    return bool(
        torch.equal(g, g[..., :1].expand_as(g))
        and torch.equal(b, b[..., :1].expand_as(b))
        and torch.equal(w, w[..., :1].expand_as(w))
        and torch.equal(b[..., 0], w[..., 0])
    )


def _load_box_kernels() -> tuple[K1Fn, K2Fn]:
    """Lazily load the compiled tcgen05 K#1/K#2 kernels (importlib keeps src typed).

    The DSL kernel modules live beside this shim but import ``cutlass`` only under a
    guarded ``try``, so they stay off the package's import graph and are pulled in
    lazily, on Blackwell hardware, exactly when dispatched. Returns ``(run_k1_incB, run_k2)``.
    """
    k1: K1Fn = importlib.import_module("lethe.kernels.cute.gdn2_bwd_dhu").run_k1_incB
    k2: K2Fn = importlib.import_module("lethe.kernels.cute.gdn2_bwd_wy").run_k2
    return k1, k2


def _load_box_kernels_cw() -> tuple[K1FnCW, K2FnCW]:
    """Lazily load the compiled channel-wise (Phase-3) tcgen05 K#1/K#2 kernels."""
    k1: K1FnCW = importlib.import_module("lethe.kernels.cute.gdn2_bwd_dhu_cw").run_k1_incB
    k2: K2FnCW = importlib.import_module("lethe.kernels.cute.gdn2_bwd_wy_cw").run_k2
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
    stage_b_closed: bool = False,
    fwd_stash: ChunkwiseForwardCW | None = None,
) -> Gdn2Grads | None:
    """Six GDN-2 gradients from the native CuTe assembly, or ``None`` if unavailable.

    Signature mirrors ``reference_gdn2_backward``. Returns ``None`` (the fallback
    contract) when the kernel is absent, the device is not Blackwell, the dtype is
    unsupported, or the tile dims do not match the kernels. The scalar-reducible regime
    (``b = w = beta``, ``g`` channel-constant) routes through the Phase-2 scalar assembly
    (d_v=64); the genuinely channel-wise regime routes through the Phase-3 channel-wise
    assembly (d_v in {64, 128}). Shapes: ``q``/``k``/``g``/``b`` [B, L, H, d_k];
    ``v``/``w``/``do`` [B, L, H, d_v]. ``stage_b_closed`` selects the chunk-local
    closed-form stage-B VJP, honored on the channel-wise route only (the closed form
    exists for the cw stage B); the scalar route ignores it.

    Double-backward (``do.requires_grad=True``) falls back to the eager path: the native
    tcgen05 kernels ingest operands via DLPack which rejects grad-requiring tensors.
    """
    if do.requires_grad:
        return None
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
        stage_b_closed=stage_b_closed,
        fwd_stash=fwd_stash,
    )


def _family_dims_ok(q: Tensor, v: Tensor) -> bool:
    """The channel-wise kernels' dim locks, shared by every family mode."""
    return bool(
        is_available(q.device)
        and q.dtype in SUPPORTED_DTYPES
        and q.shape[-1] == _KERNEL_D_K
        and q.shape[1] % _KERNEL_CHUNK == 0
        and v.shape[-1] in _KERNEL_D_V_CW
    )


def native_gla_backward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    do: Tensor,
    *,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
) -> GlaGrads | None:
    """GLA family mode through the tcgen05 kernels, or ``None`` if unavailable.

    The no-erase modes ride the skip-T fast path: K#2 never launches; K#1 runs with
    an exact-zero ``wy`` operand.
    """
    if do.requires_grad or not _family_dims_ok(q, v):
        return None
    k1_cw, _k2_cw = _load_box_kernels_cw()
    return gla_backward(q, k, v, g, do, scale=scale, use_qk_l2norm=use_qk_l2norm, k1_fn=k1_cw)


def native_la_backward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    do: Tensor,
    *,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
) -> LaGrads | None:
    """Plain linear-attention family mode, or ``None`` if unavailable."""
    if do.requires_grad or not _family_dims_ok(q, v):
        return None
    k1_cw, _k2_cw = _load_box_kernels_cw()
    return la_backward(q, k, v, do, scale=scale, use_qk_l2norm=use_qk_l2norm, k1_fn=k1_cw)


def native_ssd_backward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    do: Tensor,
    *,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
) -> SsdGrads | None:
    """SSD-class family mode (``g`` [B, L, H]), or ``None`` if unavailable."""
    if do.requires_grad or not _family_dims_ok(q, v):
        return None
    k1_cw, _k2_cw = _load_box_kernels_cw()
    return ssd_backward(q, k, v, g, do, scale=scale, use_qk_l2norm=use_qk_l2norm, k1_fn=k1_cw)


def native_kda_backward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    do: Tensor,
    *,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
) -> KdaGrads | None:
    """KDA family mode (full WY machinery, both kernels), or ``None`` if unavailable."""
    if do.requires_grad or not _family_dims_ok(q, v):
        return None
    k1_cw, k2_cw = _load_box_kernels_cw()
    return kda_backward(
        q,
        k,
        v,
        g,
        beta,
        do,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm,
        k1_fn=k1_cw,
        k2_fn=k2_cw,
    )
