"""Family oracles: independent token-serial references for the GDN-2 family gates."""

from typing import NamedTuple

import torch
from torch import Tensor

from lethe.kernels.references.gdn_backward import _l2norm


def _check_dtype(t: Tensor) -> None:
    if t.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"Expected float32/float64 input, got {t.dtype}")


def _prep_qk(
    q: Tensor, k: Tensor, scale: float | None, use_qk_l2norm: bool
) -> tuple[Tensor, Tensor]:
    s = q.shape[-1] ** -0.5 if scale is None else scale
    if use_qk_l2norm:
        q = _l2norm(q)
        k = _l2norm(k)
    return q * s, k


class GlaGrads(NamedTuple):
    """Gradients of the GLA oracle (``grad_g`` is per key-channel)."""

    grad_q: Tensor
    grad_k: Tensor
    grad_v: Tensor
    grad_g: Tensor


def reference_gla_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    *,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
) -> Tensor:
    """GLA token-serial forward. ``q``/``k``/``g`` [B, L, H, d_k]; ``v`` [B, L, H, d_v]."""
    _check_dtype(q)
    qh, kh, vh, gh = (t.transpose(1, 2) for t in (q, k, v, g))
    batch, nheads, seqlen, d_k = kh.shape
    d_v = vh.shape[-1]
    qh, kh = _prep_qk(qh, kh, scale, use_qk_l2norm)

    S = torch.zeros(batch, nheads, d_k, d_v, dtype=qh.dtype, device=qh.device)
    out = []
    for t in range(seqlen):
        S = S * gh[:, :, t].exp().unsqueeze(-1)
        S = S + kh[:, :, t].unsqueeze(-1) * vh[:, :, t].unsqueeze(-2)
        out.append((S * qh[:, :, t].unsqueeze(-1)).sum(-2))
    return torch.stack(out, dim=2).transpose(1, 2).contiguous()


def reference_gla_backward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    do: Tensor,
    *,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
) -> GlaGrads:
    """GLA backward via ``torch.autograd`` (family-gate ground truth)."""
    _check_dtype(q)
    leaves = tuple(t.detach().requires_grad_(True) for t in (q, k, v, g))
    o = reference_gla_forward(*leaves, scale=scale, use_qk_l2norm=use_qk_l2norm)
    grads = torch.autograd.grad(o, leaves, do)
    return GlaGrads(*grads)


class LaGrads(NamedTuple):
    """Gradients of the plain linear-attention oracle."""

    grad_q: Tensor
    grad_k: Tensor
    grad_v: Tensor


def reference_la_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
) -> Tensor:
    """Plain causal linear-attention token-serial forward."""
    _check_dtype(q)
    qh, kh, vh = (t.transpose(1, 2) for t in (q, k, v))
    batch, nheads, seqlen, d_k = kh.shape
    d_v = vh.shape[-1]
    qh, kh = _prep_qk(qh, kh, scale, use_qk_l2norm)

    S = torch.zeros(batch, nheads, d_k, d_v, dtype=qh.dtype, device=qh.device)
    out = []
    for t in range(seqlen):
        S = S + kh[:, :, t].unsqueeze(-1) * vh[:, :, t].unsqueeze(-2)
        out.append((S * qh[:, :, t].unsqueeze(-1)).sum(-2))
    return torch.stack(out, dim=2).transpose(1, 2).contiguous()


def reference_la_backward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    do: Tensor,
    *,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
) -> LaGrads:
    """Plain linear-attention backward via ``torch.autograd``."""
    _check_dtype(q)
    leaves = tuple(t.detach().requires_grad_(True) for t in (q, k, v))
    o = reference_la_forward(*leaves, scale=scale, use_qk_l2norm=use_qk_l2norm)
    grads = torch.autograd.grad(o, leaves, do)
    return LaGrads(*grads)


class SsdGrads(NamedTuple):
    """Gradients of the SSD-class oracle (``grad_g`` is scalar per token-head)."""

    grad_q: Tensor
    grad_k: Tensor
    grad_v: Tensor
    grad_g: Tensor


def reference_ssd_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    *,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
) -> Tensor:
    """SSD-class token-serial forward. ``g`` [B, L, H], one log-decay per token-head."""
    _check_dtype(q)
    qh, kh, vh = (t.transpose(1, 2) for t in (q, k, v))
    gh = g.transpose(1, 2)
    batch, nheads, seqlen, d_k = kh.shape
    d_v = vh.shape[-1]
    qh, kh = _prep_qk(qh, kh, scale, use_qk_l2norm)

    S = torch.zeros(batch, nheads, d_k, d_v, dtype=qh.dtype, device=qh.device)
    out = []
    for t in range(seqlen):
        S = S * gh[:, :, t].exp()[..., None, None]
        S = S + kh[:, :, t].unsqueeze(-1) * vh[:, :, t].unsqueeze(-2)
        out.append((S * qh[:, :, t].unsqueeze(-1)).sum(-2))
    return torch.stack(out, dim=2).transpose(1, 2).contiguous()


def reference_ssd_backward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    do: Tensor,
    *,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
) -> SsdGrads:
    """SSD-class backward via ``torch.autograd``."""
    _check_dtype(q)
    leaves = tuple(t.detach().requires_grad_(True) for t in (q, k, v, g))
    o = reference_ssd_forward(*leaves, scale=scale, use_qk_l2norm=use_qk_l2norm)
    grads = torch.autograd.grad(o, leaves, do)
    return SsdGrads(*grads)


class KdaGrads(NamedTuple):
    """Gradients of the KDA oracle (``grad_g`` per channel, ``grad_beta`` scalar)."""

    grad_q: Tensor
    grad_k: Tensor
    grad_v: Tensor
    grad_g: Tensor
    grad_beta: Tensor


def reference_kda_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    *,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
) -> Tensor:
    """KDA token-serial forward. ``g`` [B, L, H, d_k]; ``beta`` [B, L, H] in (0, 1)."""
    _check_dtype(q)
    qh, kh, vh, gh = (t.transpose(1, 2) for t in (q, k, v, g))
    bh = beta.transpose(1, 2)
    batch, nheads, seqlen, d_k = kh.shape
    d_v = vh.shape[-1]
    qh, kh = _prep_qk(qh, kh, scale, use_qk_l2norm)

    S = torch.zeros(batch, nheads, d_k, d_v, dtype=qh.dtype, device=qh.device)
    out = []
    for t in range(seqlen):
        k_t = kh[:, :, t]
        S = S * gh[:, :, t].exp().unsqueeze(-1)
        kS = (S * k_t.unsqueeze(-1)).sum(-2)
        v_new = bh[:, :, t].unsqueeze(-1) * (vh[:, :, t] - kS)
        S = S + k_t.unsqueeze(-1) * v_new.unsqueeze(-2)
        out.append((S * qh[:, :, t].unsqueeze(-1)).sum(-2))
    return torch.stack(out, dim=2).transpose(1, 2).contiguous()


def reference_kda_backward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    do: Tensor,
    *,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
) -> KdaGrads:
    """KDA backward via ``torch.autograd``."""
    _check_dtype(q)
    leaves = tuple(t.detach().requires_grad_(True) for t in (q, k, v, g, beta))
    o = reference_kda_forward(*leaves, scale=scale, use_qk_l2norm=use_qk_l2norm)
    grads = torch.autograd.grad(o, leaves, do)
    return KdaGrads(*grads)
