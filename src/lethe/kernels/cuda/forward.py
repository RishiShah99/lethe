"""Python entry point for the CUDA forward selective scan (Inc 1).

Mirrors ``reference_forward_chunked_scan`` semantics for fp32 CUDA tensors;
the extension is compiled on first call (see ``_loader``). Mixed precision and
the public-op dispatch wiring come with the later increments — this entry is
the parity-checkable de-risk of the scan primitive + build + wrapper.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ._loader import load_scan_extension


def cuda_forward_scan(
    u: Tensor, delta: Tensor, A: Tensor, B: Tensor, C: Tensor, D: Tensor
) -> Tensor:
    """Selective-scan forward on the CUDA kernel. fp32 CUDA inputs only.

    Args mirror ``reference_forward_chunked_scan``: ``u``/``delta`` [B, L, D],
    ``A`` [D, N] (negative log-magnitude), ``B``/``C`` [B, L, N], ``D`` [D].
    Returns ``y`` [B, L, D].
    """
    if not u.is_cuda:
        raise ValueError("cuda_forward_scan requires CUDA tensors")
    if u.dtype != torch.float32:
        raise ValueError(f"Inc 1 forward is fp32-only, got {u.dtype}")
    ext = load_scan_extension()
    y: Tensor = ext.forward_scan(
        u.contiguous(),
        delta.contiguous(),
        A.contiguous(),
        B.contiguous(),
        C.contiguous(),
        D.contiguous(),
    )
    return y


def cuda_forward_scan_2d(
    u: Tensor, delta: Tensor, A: Tensor, B: Tensor, C: Tensor, D: Tensor, *, warps: int = 16
) -> Tensor:
    """2-D forward scan (warps split d_state). fp32 CUDA inputs only.

    Same semantics as :func:`cuda_forward_scan`; ``warps`` (4/8/16/32) is the
    d_state-parallelism knob benched at Inc 2.
    """
    if not u.is_cuda:
        raise ValueError("cuda_forward_scan_2d requires CUDA tensors")
    if u.dtype != torch.float32:
        raise ValueError(f"Inc 2 forward is fp32-only, got {u.dtype}")
    ext = load_scan_extension()
    y: Tensor = ext.forward_scan_2d(
        u.contiguous(),
        delta.contiguous(),
        A.contiguous(),
        B.contiguous(),
        C.contiguous(),
        D.contiguous(),
        warps,
    )
    return y


def cuda_forward_scan_tiled(
    u: Tensor, delta: Tensor, A: Tensor, B: Tensor, C: Tensor, D: Tensor, *, items: int = 8
) -> Tensor:
    """Efficient forward: d-major layout + kNItems time-tiling. fp32 CUDA only.

    Transposes [B, L, D]/[B, L, N] inputs to the kernel's d-major [B, D, L]/
    [B, N, L] (loads coalesce on the L-fastest axis), runs the tiled scan, and
    transposes y back. ``items`` (4/8/16) is the per-thread time-tile.
    """
    if not u.is_cuda:
        raise ValueError("cuda_forward_scan_tiled requires CUDA tensors")
    if u.dtype != torch.float32:
        raise ValueError(f"tiled forward is fp32-only, got {u.dtype}")
    ext = load_scan_extension()
    u_dl = u.transpose(1, 2).contiguous()
    delta_dl = delta.transpose(1, 2).contiguous()
    b_nl = B.transpose(1, 2).contiguous()
    c_nl = C.transpose(1, 2).contiguous()
    y_dl: Tensor = ext.forward_scan_tiled(
        u_dl, delta_dl, A.contiguous(), b_nl, c_nl, D.contiguous(), items
    )
    return y_dl.transpose(1, 2).contiguous()
