"""Fused Mamba block forward: conv1d + SiLU + selective scan + RMSNorm (eager).

The input arrives already left-padded with conv_kernel_size - 1 zeros, so
the depthwise conv is a VALID convolution. fp16/bf16 inputs are upcast
once to float32 (including the RMSNorm sum-of-squares) and rounded once
at the output.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def fused_block_forward(
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
    out_dtype = x.dtype
    if out_dtype in (torch.float16, torch.bfloat16):
        x, conv_weight, conv_bias, delta, A, B, C, D, norm_weight = (
            t.to(torch.float32) for t in (x, conv_weight, conv_bias, delta, A, B, C, D, norm_weight)
        )

    batch, _seq_len, d_model = x.shape
    n_state = A.shape[1]

    x_t = x.transpose(1, 2)
    conv_out = F.conv1d(x_t, conv_weight, conv_bias, groups=d_model)
    conv_out = conv_out.transpose(1, 2)
    z = F.silu(conv_out)

    l_out = z.shape[1]
    delta_bar = F.softplus(delta)
    a_bar = torch.exp(delta_bar.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
    b_bar = delta_bar.unsqueeze(-1) * B.unsqueeze(2)

    y_scan = torch.empty_like(z)
    h = torch.zeros(batch, d_model, n_state, dtype=z.dtype, device=z.device)
    for t in range(l_out):
        h = a_bar[:, t, :, :] * h + b_bar[:, t, :, :] * z[:, t, :].unsqueeze(-1)
        y_scan[:, t, :] = (h * C[:, t, :].unsqueeze(1)).sum(-1) + D * z[:, t, :]

    rms = y_scan.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    y = y_scan / rms * norm_weight
    return y.to(out_dtype)
