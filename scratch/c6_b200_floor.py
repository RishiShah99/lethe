"""B200 re-measurement of the fused-backward PRC-02 floors at the gate shape.

CPU calibration (c6_prc02_floor.py) measured the eager backward's honest
floor; this records the Triton pipeline's actual honest floor (fp16 inputs
vs fp32 reference grads) and the fp16-scan-state cheat's autograd, per
gradient view, as fractions of output scale against the gate's per-view
scale-aware unit atols. PRC-02 runs saturation-free, so the aux here does
too. Box usage:
fleet run "bash scratch/detach.sh uv run python scratch/c6_b200_floor.py"
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from lethe.kernels.ops import fused_block_backward
from lethe.kernels.references.fused_block_backward import (
    FusedBlockGrads,
    reference_fused_block_backward,
)
from lethe.verifier.op_harness import (
    FUSED_BWD_GATE_OVERRIDES,
    FUSED_CONV_K,
    SCAN_N_STATE,
    _fused_bwd_aux,
)

UNIT_ATOLS = {
    field: FUSED_BWD_GATE_OVERRIDES[field]["gate_prc_02_mixed_precision_accumulation"]["atol"]
    for field in FusedBlockGrads._fields
}


def fp16_state_fused(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    norm_weight: torch.Tensor,
) -> torch.Tensor:
    """C5's cheat: conv/SiLU/norm in fp32 but the scan state carried in fp16."""
    xf, wf, bf, dltf, af, bpf, cpf, dsf, nwf = (
        t.float() for t in (x, conv_weight, conv_bias, delta, A, B, C, D, norm_weight)
    )
    d_model = xf.shape[-1]
    conv = F.conv1d(xf.transpose(1, 2), wf, bf, groups=d_model).transpose(1, 2)
    z = F.silu(conv)
    delta_bar = F.softplus(dltf)
    a_bar = torch.exp(delta_bar.unsqueeze(-1) * af.unsqueeze(0).unsqueeze(0))
    b_bar = delta_bar.unsqueeze(-1) * bpf.unsqueeze(2)
    batch, l_out, _ = z.shape
    h = torch.zeros(batch, d_model, af.shape[1], dtype=torch.float16, device=xf.device)
    ys = []
    for t in range(l_out):
        bu = b_bar[:, t] * z[:, t].unsqueeze(-1)
        h = (a_bar[:, t].half() * h + bu.half()).half()
        ys.append((h.float() * cpf[:, t].unsqueeze(1)).sum(-1) + dsf * z[:, t])
    y_scan = torch.stack(ys, dim=1)
    rms = y_scan.pow(2).mean(dim=-1, keepdim=True).add(1e-5).sqrt()
    return (y_scan / rms * nwf).to(x.dtype)


def cheat_backward(*inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
    args, dy = inputs[:9], inputs[9]
    with torch.enable_grad():
        leaves = [t.detach().requires_grad_(True) for t in args]
        y = fp16_state_fused(*leaves)
        return torch.autograd.grad(y, leaves, dy)


def main() -> None:
    dev = torch.device("cuda")
    batch, seqlen, d_model = 1, 4096, 32
    fields = FusedBlockGrads._fields
    worst_hon = dict.fromkeys(fields, 0.0)
    worst_cheat = dict.fromkeys(fields, 1e9)
    for seed in range(3):
        torch.manual_seed(seed)
        dy32 = torch.randn(batch, seqlen, d_model, device=dev)
        dy16 = dy32.to(torch.float16)
        aux32 = _fused_bwd_aux(
            batch, seqlen, d_model, SCAN_N_STATE, FUSED_CONV_K, dev, torch.float32, saturate=False
        )
        aux16 = _fused_bwd_aux(
            batch, seqlen, d_model, SCAN_N_STATE, FUSED_CONV_K, dev, torch.float16, saturate=False
        )
        pad32 = F.pad(aux32[0], (0, 0, FUSED_CONV_K - 1, 0))
        pad16 = F.pad(aux16[0], (0, 0, FUSED_CONV_K - 1, 0))
        ref = reference_fused_block_backward(pad32, *aux32[1:], dy32, chunk_size=8)
        hon = fused_block_backward(pad16, *aux16[1:], dy16, chunk_size=8)
        cheat = cheat_backward(pad16, *aux16[1:], dy16)
        for field, r, h, c in zip(fields, ref, hon, cheat, strict=True):
            scale = max(1.0, r.abs().max().item())
            hon_rel = (h.float() - r).abs().max().item() / scale
            cheat_rel = (c.float() - r).abs().max().item() / scale
            worst_hon[field] = max(worst_hon[field], hon_rel)
            worst_cheat[field] = min(worst_cheat[field], cheat_rel)
            print(
                f"seed={seed} {field:18s} scale={scale:9.3e} "
                f"kernel={hon_rel:9.3e} cheat={cheat_rel:9.3e}"
            )
    print()
    for field in fields:
        atol = UNIT_ATOLS[field]
        under = atol / max(worst_hon[field], 1e-12)
        over = worst_cheat[field] / atol
        print(
            f"--> {field:18s} kernel<= {worst_hon[field]:9.3e}  "
            f"cheat>= {worst_cheat[field]:9.3e}  atol={atol:.0e} "
            f"({under:5.1f}x under / {over:5.1f}x over)"
        )


if __name__ == "__main__":
    main()
