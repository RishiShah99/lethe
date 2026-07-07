"""Measure PRC-02 floors for the nine fused-backward views at (1, 4096, 32).

Honest = the eager backward (fp32 compute from fp16-rounded inputs) vs the
fp32 reference grads; cheat = autograd through the C5 fp16-scan-state
forward cheat (its backward carry re-rounds to fp16 every step, exactly
the pathology PRC-02 exists to catch). Prints per-view scale-normalised
max_err over 3 gate draws — the numbers behind FUSED_BWD_GATE_OVERRIDES'
PRC-02 atols and the discrimination pin.

PRC-02 runs saturation-free (the harness rerun), so the aux here does too.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from lethe.kernels.ops import fused_block_backward
from lethe.kernels.references.fused_block_backward import (
    FusedBlockGrads,
    reference_fused_block_backward,
)
from lethe.verifier.op_harness import FUSED_CONV_K, SCAN_N_STATE, _fused_bwd_aux


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
    h = torch.zeros(batch, d_model, af.shape[1], dtype=torch.float16)
    ys = []
    for t in range(l_out):
        bu = b_bar[:, t] * z[:, t].unsqueeze(-1)
        h = (a_bar[:, t].half() * h + bu.half()).half()
        ys.append((h.float() * cpf[:, t].unsqueeze(1)).sum(-1) + dsf * z[:, t])
    y_scan = torch.stack(ys, dim=1)
    rms = y_scan.pow(2).mean(dim=-1, keepdim=True).add(1e-5).sqrt()
    return (y_scan / rms * nwf).to(x.dtype)


def cheat_backward(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    norm_weight: torch.Tensor,
    dy: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    inputs = (x, conv_weight, conv_bias, delta, A, B, C, D, norm_weight)
    with torch.enable_grad():
        leaves = [t.detach().requires_grad_(True) for t in inputs]
        y = fp16_state_fused(*leaves)
        return torch.autograd.grad(y, leaves, dy)


def main() -> None:
    batch, seqlen, d_model = 1, 4096, 32
    fields = FusedBlockGrads._fields
    worst_hon = dict.fromkeys(fields, 0.0)
    worst_cheat = dict.fromkeys(fields, 1e9)
    for seed in range(3):
        torch.manual_seed(seed)
        dy32 = torch.randn(batch, seqlen, d_model)
        dy16 = dy32.to(torch.float16)
        aux32 = _fused_bwd_aux(
            batch,
            seqlen,
            d_model,
            SCAN_N_STATE,
            FUSED_CONV_K,
            torch.device("cpu"),
            torch.float32,
            saturate=False,
        )
        aux16 = _fused_bwd_aux(
            batch,
            seqlen,
            d_model,
            SCAN_N_STATE,
            FUSED_CONV_K,
            torch.device("cpu"),
            torch.float16,
            saturate=False,
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
                f"honest={hon_rel:9.3e} cheat={cheat_rel:9.3e}"
            )
    print()
    for field in fields:
        sep = worst_cheat[field] / max(worst_hon[field], 1e-12)
        print(
            f"--> {field:18s} honest<= {worst_hon[field]:9.3e}  "
            f"cheat>= {worst_cheat[field]:9.3e}  ({sep:6.1f}x)"
        )


if __name__ == "__main__":
    main()
