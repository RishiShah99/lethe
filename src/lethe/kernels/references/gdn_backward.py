"""Gated DeltaNet-2 (GDN-2) chunkwise recurrence: forward reference + autograd backward."""

from typing import NamedTuple

import torch
from torch import Tensor


def _l2norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)


def reference_gdn2_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    b: Tensor,
    w: Tensor,
    *,
    scale: float | None = None,
    initial_state: Tensor | None = None,
    use_qk_l2norm: bool = True,
) -> Tensor:
    """GDN-2 chunkwise recurrence forward (token-serial reference)."""
    if q.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"Expected float32/float64 input, got {q.dtype}")

    qh, kh, vh, gh, bh, wh = (t.transpose(1, 2) for t in (q, k, v, g, b, w))
    batch, nheads, seqlen, d_k = kh.shape
    d_v = vh.shape[-1]
    s = d_k**-0.5 if scale is None else scale

    if use_qk_l2norm:
        qh = _l2norm(qh)
        kh = _l2norm(kh)
    qh = qh * s

    if initial_state is not None:
        S = initial_state.to(qh.dtype)
    else:
        S = torch.zeros(batch, nheads, d_k, d_v, dtype=qh.dtype, device=qh.device)

    out = []
    for t in range(seqlen):
        g_t, k_t, b_t = gh[:, :, t], kh[:, :, t], bh[:, :, t]
        v_t, w_t, q_t = vh[:, :, t], wh[:, :, t], qh[:, :, t]

        S = S * g_t.exp().unsqueeze(-1)
        erase = (S * (b_t * k_t).unsqueeze(-1)).sum(-2)
        v_new = w_t * v_t - erase
        S = S + k_t.unsqueeze(-1) * v_new.unsqueeze(-2)
        out.append((S * q_t.unsqueeze(-1)).sum(-2))

    return torch.stack(out, dim=2).transpose(1, 2).contiguous()


class Gdn2Grads(NamedTuple):
    """Gradient bundle returned by the GDN-2 backward reference."""

    grad_q: Tensor
    grad_k: Tensor
    grad_v: Tensor
    grad_g: Tensor
    grad_b: Tensor
    grad_w: Tensor
    grad_initial_state: Tensor | None


def reference_gdn2_backward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    b: Tensor,
    w: Tensor,
    do: Tensor,
    *,
    scale: float | None = None,
    initial_state: Tensor | None = None,
    use_qk_l2norm: bool = True,
) -> Gdn2Grads:
    """GDN-2 backward via ``torch.autograd`` (ground truth for the kernel)."""
    if q.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"Expected float32/float64 input, got {q.dtype}")

    q_l = q.detach().requires_grad_(True)
    k_l = k.detach().requires_grad_(True)
    v_l = v.detach().requires_grad_(True)
    g_l = g.detach().requires_grad_(True)
    b_l = b.detach().requires_grad_(True)
    w_l = w.detach().requires_grad_(True)
    leaves = [q_l, k_l, v_l, g_l, b_l, w_l]

    s_l = None
    if initial_state is not None:
        s_l = initial_state.detach().requires_grad_(True)
        leaves.append(s_l)

    o = reference_gdn2_forward(
        q_l,
        k_l,
        v_l,
        g_l,
        b_l,
        w_l,
        scale=scale,
        initial_state=s_l,
        use_qk_l2norm=use_qk_l2norm,
    )

    grads = torch.autograd.grad(outputs=o, inputs=tuple(leaves), grad_outputs=do)

    return Gdn2Grads(
        grad_q=grads[0],
        grad_k=grads[1],
        grad_v=grads[2],
        grad_g=grads[3],
        grad_b=grads[4],
        grad_w=grads[5],
        grad_initial_state=grads[6] if s_l is not None else None,
    )
