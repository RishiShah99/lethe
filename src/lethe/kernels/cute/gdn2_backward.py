"""Native Blackwell (sm_100) CuTe/tcgen05 GDN-2 training backward, dispatch shim."""

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
# Channel-wise kernels route through the same (128,64,128) config; N-tiling lets d_v be 64 or 128.
_KERNEL_D_V_CW: tuple[int, ...] = (64, 128)


def is_available(device: torch.device | None = None) -> bool:
    """True iff the compiled sm_100 assembly is validated AND ``device`` is Blackwell."""
    if not torch.cuda.is_available():
        return False
    if device is not None and device.type != "cuda":
        return False
    major, _minor = torch.cuda.get_device_capability(device)
    return major == 10


def _is_scalar_reducible(g: Tensor, b: Tensor, w: Tensor) -> bool:
    """True iff ``g`` is channel-constant and ``b == w == beta·1`` (the Phase-2 regime)."""
    return bool(
        torch.equal(g, g[..., :1].expand_as(g))
        and torch.equal(b, b[..., :1].expand_as(b))
        and torch.equal(w, w[..., :1].expand_as(w))
        and torch.equal(b[..., 0], w[..., 0])
    )


def _load_box_kernels() -> tuple[K1Fn, K2Fn]:
    """Lazily load the compiled tcgen05 K#1/K#2 kernels (importlib keeps src typed)."""
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
    """Six GDN-2 gradients from the native CuTe assembly, or ``None`` if unavailable."""
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
    """GLA family mode through the tcgen05 kernels, or ``None`` if unavailable."""
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
