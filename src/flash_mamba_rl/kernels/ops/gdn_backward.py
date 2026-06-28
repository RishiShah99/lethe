"""GDN-2 chunkwise backward — candidate op (oracle-faithful eager path).

Drop-in for ``reference_gdn2_backward`` with the same ``(q, k, v, g, b, w, do)``
signature, widened to half dtypes:

- fp32/fp64 -> autograd straight through ``reference_gdn2_forward`` (gradients
  bitwise-equal to the oracle).
- fp16/bf16 -> leaves upcast to fp32, each gradient rounds once at the end (the
  mixed-precision contract PRC-02 measures). When ``do.requires_grad`` the eager
  path builds the double-backward graph (``create_graph``) for CMP-02's gradcheck.

The native Blackwell tcgen05/TMEM kernel (Phase 2/3) dispatches from here on CUDA,
exactly as ``mimo_backward`` dispatches to its Triton kernel. Until it exists this
is the eager path only — the harness verifies *this* against the oracle so the
wiring is proven before the kernel lands.
"""

from __future__ import annotations

import torch
from torch import Tensor

from flash_mamba_rl.kernels.references.gdn_backward import Gdn2Grads, reference_gdn2_forward


def gdn2_backward(
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
) -> Gdn2Grads:
    """GDN-2 backward. Returns ``Gdn2Grads`` (``grad_initial_state`` is ``None``).

    Args/semantics mirror ``reference_gdn2_backward``: ``q``/``k``/``g``/``b``
    [B, L, H, d_k], ``v``/``w``/``do`` [B, L, H, d_v]; each returned gradient
    matches its input's shape and dtype.
    """
    compute_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    inputs = (q, k, v, g, b, w)
    create_graph = do.requires_grad
    with torch.enable_grad():
        leaves = [t.detach().to(compute_dtype).requires_grad_(True) for t in inputs]
        o = reference_gdn2_forward(*leaves, scale=scale, use_qk_l2norm=use_qk_l2norm)
        grads = torch.autograd.grad(o, leaves, do.to(compute_dtype), create_graph=create_graph)
    dq, dk, dv, dg, db, dw = (grad.to(t.dtype) for grad, t in zip(grads, inputs, strict=True))
    return Gdn2Grads(dq, dk, dv, dg, db, dw, None)
