"""Native Blackwell (sm_100) CuTe/tcgen05 GDN-2 training backward — dispatch shim.

Phase 2/3 (PROJECT_PLAN). Two hard kernels sit behind this entry point:

- the reverse inter-chunk state recurrence — ``dS`` [d_k, d_v] fp32 accumulator
  resident in TMEM across the reverse scan (the clean whitespace cuLA leaves in
  Triton);
- the WY / triangular-inverse VJP — ``dA <- strict_tril(A @ (dA @ A))`` in fp32,
  channel-wise in GDN-2.

Until the kernel is built and loadable on an sm_100 device, ``native_gdn2_backward``
returns ``None`` and the caller (``ops.gdn_backward.gdn2_backward``) falls back to
the oracle-faithful eager path. The verifier grades whichever path actually runs,
so the fallback is contract-correct today and the kernel slots in transparently
once Phase 2 lands. The loader is intentionally absent here — the C++-CuTe vs
Python-CuTe-DSL choice is resolved in Phase 2 before the loader is written.
"""

from __future__ import annotations

import torch
from torch import Tensor

from flash_mamba_rl.kernels.references.gdn_backward import Gdn2Grads

SUPPORTED_DTYPES: tuple[torch.dtype, ...] = (torch.bfloat16, torch.float16, torch.float32)


def is_available(device: torch.device | None = None) -> bool:
    """True iff the compiled sm_100 kernel is present AND ``device`` is Blackwell.

    Scaffold: always ``False`` until Phase 2 lands the kernel + its loader.
    """
    return False


def native_gdn2_backward(
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
) -> Gdn2Grads | None:
    """Six GDN-2 gradients from the native CuTe kernel, or ``None`` if unavailable.

    Signature mirrors ``reference_gdn2_backward``. Returning ``None`` (not raising)
    is the contract: the caller falls back to the eager path. Shapes: ``q``/``k``/
    ``g``/``b`` [B, L, H, d_k]; ``v``/``w``/``do`` [B, L, H, d_v].
    """
    if not is_available(q.device):
        return None
    raise NotImplementedError("native GDN-2 CuTe backward not yet built (Phase 2)")
