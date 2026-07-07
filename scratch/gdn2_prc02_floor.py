"""Calibrate GDN-2 PRC-02: honest-vs-fp16-state-cheat error per grad view.

CPU, fp16 inputs at (1, L, 32). Prints, per field, honest and cheat max_err
(absolute and as fraction of output scale) so the unit atol can sit between.
"""

import torch
from tests.test_op_harness_gdn2 import _bf16_state_gdn2_bwd

from lethe.kernels.ops import gdn2_backward
from lethe.verifier.op_harness import (
    GDN2_BWD_GRAD_FIELDS,
    gdn2_bwd_candidate_adapter,
    gdn2_bwd_reference_adapter,
)

L = 4096  # the PRC-02 gate stress shape
DRAWS = 2
shape = (1, L, 32)

for field in GDN2_BWD_GRAD_FIELDS:
    ref = gdn2_bwd_reference_adapter(field)
    honest = gdn2_bwd_candidate_adapter(gdn2_backward, field)
    cheat = gdn2_bwd_candidate_adapter(_bf16_state_gdn2_bwd, field)

    h_fracs, c_fracs = [], []
    for d in range(DRAWS):
        torch.manual_seed(1000 + d)
        t32 = torch.randn(shape, dtype=torch.float32)
        t16 = t32.to(torch.float16)
        r = ref(t32).float()
        scale = max(1.0, r[torch.isfinite(r)].abs().max().item())
        h_fracs.append((honest(t16).float() - r).abs().max().item() / scale)
        c_fracs.append((cheat(t16).float() - r).abs().max().item() / scale)
    print(
        f"{field:8s} honest max={max(h_fracs):.3e}/s  bf16-cheat min={min(c_fracs):.3e}/s  "
        f"margin={min(c_fracs) / max(h_fracs):5.1f}x"
    )
