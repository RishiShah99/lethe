"""Mamba-3 fused-block forward pass (un-fused PyTorch reference).

Implements the full Mamba block as a composition of:
  1. Depthwise conv1d  (causal, groups=D)
  2. SiLU activation
  3. Selective SSM scan  (reuses reference_forward_chunked_scan)
  4. RMSNorm

No fusion, no Triton; correctness oracle only.

Reference: Mamba-3 (Lahoti, Li, et al., ICLR 2026), building on
Mamba-1 (Gu & Dao, ICLR 2024, arXiv:2312.00752).
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from .forward_chunked_scan import reference_forward_chunked_scan


def _rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Element-wise RMSNorm: x / rms(x) * weight."""
    rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    return x / rms * weight


def reference_fused_block_forward(
    x: Tensor,
    conv_weight: Tensor,
    conv_bias: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor,
    norm_weight: Tensor,
    *,
    conv_kernel_size: int = 4,
    eps: float = 1e-5,
    chunk_size: int = 64,
) -> Tensor:
    """Mamba fused-block forward: conv1d → SiLU → selective scan → RMSNorm.

    This is an *un-fused* reference implementation using plain PyTorch ops.
    It is mathematically equivalent to the fused Triton kernel but makes no
    attempt at efficiency.

    Pipeline
    --------
    1. Causal depthwise conv1d over the sequence dimension (groups = D,
       kernel ``conv_kernel_size``, no padding inserted here; caller must
       pre-pad or pass ``x`` with causal left-padding of
       ``conv_kernel_size - 1`` zeros along L).
    2. SiLU activation applied element-wise.
    3. ``reference_forward_chunked_scan`` with the SSM parameters.
    4. RMSNorm along the D dimension with learned ``norm_weight``.

    Args:
        x:               Input, shape [B, L, D], float32.  Should carry
                         ``conv_kernel_size - 1`` left-padding in L if causal
                         masking is needed; this function does NOT pad.
        conv_weight:     Depthwise conv kernel, shape [D, 1, conv_kernel_size],
                         float32.
        conv_bias:       Depthwise conv bias, shape [D], float32.
        delta:           SSM timescale (pre-softplus), shape [B, L_out, D]
                         where L_out = L - (conv_kernel_size - 1).
        A:               Log-neg SSM matrix, shape [D, N], float32.
        B:               Input projection, shape [B, L_out, N], float32.
        C:               Output projection, shape [B, L_out, N], float32.
        D:               Skip weight, shape [D], float32.
        norm_weight:     RMSNorm gain, shape [D], float32.
        conv_kernel_size: Width of the depthwise causal conv.  Default 4.
        eps:             RMSNorm numerical stability epsilon.
        chunk_size:      Forwarded to the chunked scan.

    Returns:
        y: Output tensor, shape [B, L_out, D], float32, where
           L_out = L - (conv_kernel_size - 1).

    Raises:
        ValueError: If dtypes are not float32 or shapes are inconsistent,
            including x's channel dim disagreeing with ``conv_weight``'s.
    """
    if x.dtype != torch.float32:
        raise ValueError(f"Expected float32, got x.dtype={x.dtype}")

    _batch, _seq_len, d_model = x.shape
    # Depthwise (groups=D) conv requires x channels == D. torch would instead
    # silently reinterpret a mismatch as a grouped conv (upweighting the
    # channels), not this block's math, so reject it, matching the
    # channel-rigid contract the sibling references raise on structurally.
    if d_model != conv_weight.shape[0]:
        raise ValueError(
            f"channel mismatch: x has {d_model} channels, conv_weight has {conv_weight.shape[0]}"
        )

    # --- 1. Causal depthwise conv1d ---
    # F.conv1d expects [B, C, L]; conv_weight is [D, 1, K] for groups=D.
    x_t = x.transpose(1, 2)  # [B, D, L]
    conv_out = F.conv1d(x_t, conv_weight, conv_bias, groups=d_model)
    # conv_out: [B, D, L - (K-1)] (valid convolution, K = conv_kernel_size)
    conv_out = conv_out.transpose(1, 2)  # [B, L_out, D]

    # --- 2. SiLU ---
    z = F.silu(conv_out)  # [B, L_out, D]

    # --- 3. Selective scan ---
    y_scan = reference_forward_chunked_scan(
        z, delta, A, B, C, D, chunk_size=chunk_size
    )  # [B, L_out, D]

    # --- 4. RMSNorm ---
    y = _rms_norm(y_scan, norm_weight, eps)

    return y
