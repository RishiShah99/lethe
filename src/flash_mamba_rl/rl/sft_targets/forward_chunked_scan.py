"""Mamba SISO selective-scan forward pass (eager PyTorch).

fp16/bf16 inputs are upcast once to float32, computed with float32 state,
and rounded once at the output. Non-finites propagate through the
recurrence untouched; the reduction order is fixed, so repeated calls are
byte-identical.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def forward_chunked_scan(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor,
    *,
    chunk_size: int = 64,
) -> Tensor:
    out_dtype = u.dtype
    if out_dtype in (torch.float16, torch.bfloat16):
        u, delta, A, B, C, D = (t.to(torch.float32) for t in (u, delta, A, B, C, D))

    batch, seq_len, d_model = u.shape
    n_state = A.shape[1]

    delta_bar = F.softplus(delta)
    a_bar = torch.exp(delta_bar.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
    b_bar = delta_bar.unsqueeze(-1) * B.unsqueeze(2)

    y = torch.empty_like(u)
    h = torch.zeros(batch, d_model, n_state, dtype=u.dtype, device=u.device)
    for t in range(seq_len):
        h = a_bar[:, t, :, :] * h + b_bar[:, t, :, :] * u[:, t, :].unsqueeze(-1)
        y[:, t, :] = (h * C[:, t, :].unsqueeze(1)).sum(-1) + D * u[:, t, :]

    return y.to(out_dtype)
