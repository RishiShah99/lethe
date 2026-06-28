"""Gated DeltaNet-2 (GDN-2) chunkwise recurrence: forward reference + autograd backward.

Oracle for the native Blackwell GDN-2 training backward. The forward is an explicit
token-serial scan (slow, correct); the backward delegates to ``torch.autograd`` —
correctness by construction, not by hand-derived formulae.

RECURRENCE (locked from NVIDIA's own GDN-2 source, not the paper)
-----------------------------------------------------------------
State ``S in R^{d_k x d_v}``. Per token, with ``*``/``⊙`` elementwise:

    S      <- Diag(exp(g)) @ S            # channel-wise decay, g on the key axis
    v_new   = (w ⊙ v) - (b ⊙ k)^T @ S     # gated write minus gated read   (in R^{d_v})
    S      <- S + k ⊗ v_new               # rank-one write
    o       = S^T @ q                      # output read                    (in R^{d_v})

``q``/``k`` are L2-normalized (eps 1e-6) then ``q`` is scaled by ``1/sqrt(d_k)``.
``g`` is channel-wise log-decay on the key axis; ``b in [0,1]^{d_k}`` is the erase
gate (key axis); ``w in [0,1]^{d_v}`` is the write gate (value axis). The read uses
the post-write state (matches fla naive + the GDN-2 recurrent kernel exactly).

Source of truth: ``NVlabs/GatedDeltaNet-2`` @ 95709fc ``lit_gpt/gdn2_ops/`` — the
``fused_recurrent_gdn2`` docstring states the four-line recurrence above; this oracle
reimplements that math (NC license: math only, never copy code). Cross-checked against
``fla`` ``naive_recurrent_gated_delta_rule`` for the scalar reduction.

GDN-2 -> KDA SELF-CONSISTENCY (built-in correctness test)
---------------------------------------------------------
Setting ``b = w = beta · 1`` gives ``v_new = beta·v - beta·(k^T S) = beta·(v - k^T S)``,
which is exactly the scalar gated delta rule (fla ``naive_recurrent_gated_delta_rule``).
``reference_gdn2_forward`` with constant ``b``/``w`` must equal that scalar path bit-for-bit
(machine precision). This is the cheapest GO/NO-GO check for any GDN-2 kernel.

SHAPES (crown target H == HV, i.e. no GVA; see PROJECT_PLAN §6)
--------------------------------------------------------------
    q, k, g, b : (batch, seqlen, nheads, d_k)
    v, w       : (batch, seqlen, nheads, d_v)
    o          : (batch, seqlen, nheads, d_v)
    initial_state (optional) : (batch, nheads, d_k, d_v)

BACKWARD STRUCTURE (for the Phase-2/3 kernel implementor; the oracle uses autograd)
-----------------------------------------------------------------------------------
The kernel must produce ``dq, dk, dv, dg, db, dw`` (and ``dh0`` if an initial state is
given). The hard part is a reverse-time scan: ``S_t`` feeds ``S_{t+1}`` through BOTH the
decay carry ``Diag(exp(g_{t+1})) S_t`` AND the erase term
``-k_{t+1} [(b⊙k)_{t+1}^T Diag(exp(g_{t+1})) S_t]^T`` (delta-rule coupling), so the state
gradient ``dS`` accumulates backward with a rank-one correction per step — the reverse
inter-chunk state recurrence (``chunk_gated_delta_rule_bwd_dhu`` in fla, the kernel even
cuLA leaves in Triton). The read ``o_t = S_t^T q_t`` contributes ``dq_t = S_t @ do_t`` and a
``q_t ⊗ do_t`` term into ``dS_t``. The triangular-inverse / WY VJP handles the intra-chunk
``(I+L)^{-1}`` dependence. Study (don't copy): fla ``chunk.py`` (7-stage algo),
``chunk_gated_delta_rule_bwd_dhu``, ``prepare_wy_repr_bwd``. This oracle is the ground truth
those kernels are validated against.
"""

from typing import NamedTuple

import torch
from torch import Tensor


def _l2norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


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
    """GDN-2 chunkwise recurrence forward (token-serial reference).

    Shape contracts
    ---------------
    q, k, g, b : (batch, seqlen, nheads, d_k)   float32/float64
    v, w       : (batch, seqlen, nheads, d_v)   float32/float64
    g          : channel-wise log-decay (key axis); b, w : gates in [0, 1]
    scale      : query scale, defaults to ``d_k ** -0.5``
    initial_state : (batch, nheads, d_k, d_v), optional

    Returns
    -------
    o : (batch, seqlen, nheads, d_v)

    Notes
    -----
    * Explicit Python scan over t (oracle, not kernel). State reassigned
      functionally each step (no in-place mutation) so autograd is clean.
    * float64 accepted alongside float32 so ``torch.autograd.gradcheck`` can
      validate this exact function; half precision is rejected.
    """
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


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------


class Gdn2Grads(NamedTuple):
    """Gradient bundle returned by the GDN-2 backward reference.

    ``grad_initial_state`` is ``None`` when no initial state was supplied.
    """

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
    """GDN-2 backward via ``torch.autograd`` (ground truth for the kernel).

    Wraps ``reference_gdn2_forward`` with ``requires_grad=True`` leaves, calls
    ``torch.autograd.grad`` with the upstream gradient ``do``, and returns the
    six input gradients (plus the initial-state gradient when applicable). The
    analytic backward structure is documented in the module docstring for the
    kernel implementor; this oracle is correct by construction.

    Shape contracts match ``reference_gdn2_forward``; ``do`` matches ``o``
    ``(batch, seqlen, nheads, d_v)``.
    """
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
