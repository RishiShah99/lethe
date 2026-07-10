"""Mamba-3 fused-block forward pass (un-fused PyTorch reference)."""

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
    """Mamba fused-block forward: conv1d → SiLU → selective scan → RMSNorm."""
    if x.dtype != torch.float32:
        raise ValueError(f"Expected float32, got x.dtype={x.dtype}")

    _batch, _seq_len, d_model = x.shape
    # Depthwise (groups=D) conv requires x channels == D.
    if d_model != conv_weight.shape[0]:
        raise ValueError(
            f"channel mismatch: x has {d_model} channels, conv_weight has {conv_weight.shape[0]}"
        )

    # --- 1. Causal depthwise conv1d ---
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
