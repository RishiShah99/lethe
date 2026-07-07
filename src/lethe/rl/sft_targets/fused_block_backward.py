"""Fused Mamba block backward via autograd (eager PyTorch).

The full forward (depthwise causal conv + SiLU + selective scan + RMSNorm)
is rebuilt with leaf tensors and differentiated with torch.autograd.grad,
so all nine gradients' non-finite dataflow matches autograd's grouping.
The conv is an explicit K-term shifted sum rather than a conv primitive:
its autograd backward is then composed purely of deterministic slice/sum
ops, keeping every gradient byte-identical across calls. fp16/bf16 inputs
are upcast once to float32, differentiated in float32, and each gradient
is rounded once at return.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def _block_forward(
    x: Tensor,
    conv_weight: Tensor,
    conv_bias: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor,
    norm_weight: Tensor,
    conv_kernel_size: int,
    eps: float,
) -> Tensor:
    batch, seq_len, d_model = x.shape
    n_state = A.shape[1]
    l_out = seq_len - (conv_kernel_size - 1)

    conv_out = x[:, 0:l_out, :] * conv_weight[:, 0, 0]
    for k in range(1, conv_kernel_size):
        conv_out = conv_out + x[:, k : k + l_out, :] * conv_weight[:, 0, k]
    conv_out = conv_out + conv_bias
    z = F.silu(conv_out)

    delta_bar = F.softplus(delta)
    a_bar = torch.exp(delta_bar.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
    b_bar = delta_bar.unsqueeze(-1) * B.unsqueeze(2)

    y_scan = torch.empty_like(z)
    h = torch.zeros(batch, d_model, n_state, dtype=z.dtype, device=z.device)
    for t in range(l_out):
        h = a_bar[:, t, :, :] * h + b_bar[:, t, :, :] * z[:, t, :].unsqueeze(-1)
        y_scan[:, t, :] = (h * C[:, t, :].unsqueeze(1)).sum(-1) + D * z[:, t, :]

    rms = y_scan.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    return y_scan / rms * norm_weight


def fused_block_backward(
    x: Tensor,
    conv_weight: Tensor,
    conv_bias: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor,
    norm_weight: Tensor,
    dy: Tensor,
    *,
    conv_kernel_size: int = 4,
    eps: float = 1e-5,
    chunk_size: int = 64,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    # Same divisibility contract as the reference + triton sibling — a warm-start
    # target must not teach a looser contract than ground truth.
    l_out = x.shape[1] - (conv_kernel_size - 1)
    if l_out % chunk_size != 0:
        raise ValueError(f"output length {l_out} must be divisible by chunk_size {chunk_size}")
    out_dtype = x.dtype
    if out_dtype in (torch.float16, torch.bfloat16):
        x, conv_weight, conv_bias, delta, A, B, C, D, norm_weight, dy = (
            t.to(torch.float32)
            for t in (x, conv_weight, conv_bias, delta, A, B, C, D, norm_weight, dy)
        )

    x_l = x.detach().requires_grad_(True)
    cw_l = conv_weight.detach().requires_grad_(True)
    cb_l = conv_bias.detach().requires_grad_(True)
    delta_l = delta.detach().requires_grad_(True)
    a_l = A.detach().requires_grad_(True)
    b_l = B.detach().requires_grad_(True)
    c_l = C.detach().requires_grad_(True)
    d_l = D.detach().requires_grad_(True)
    nw_l = norm_weight.detach().requires_grad_(True)

    y = _block_forward(x_l, cw_l, cb_l, delta_l, a_l, b_l, c_l, d_l, nw_l, conv_kernel_size, eps)
    grads = torch.autograd.grad(
        outputs=y,
        inputs=(x_l, cw_l, cb_l, delta_l, a_l, b_l, c_l, d_l, nw_l),
        grad_outputs=dy,
    )
    return (
        grads[0].to(out_dtype),
        grads[1].to(out_dtype),
        grads[2].to(out_dtype),
        grads[3].to(out_dtype),
        grads[4].to(out_dtype),
        grads[5].to(out_dtype),
        grads[6].to(out_dtype),
        grads[7].to(out_dtype),
        grads[8].to(out_dtype),
    )
