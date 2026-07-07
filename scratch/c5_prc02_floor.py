"""Measure PRC-02 floors for the fused-block view at the gate stress shape.

Honest = the eager op (fp32 compute from fp16-rounded inputs) vs the fp32
reference; cheats = the same composition with the scan state carried in
fp16 (mild) and additionally an fp16 readout dot (harsh). Prints per-shape
raw and scale-normalised max_err over several gate draws — the numbers
behind FUSED_GATE_OVERRIDES' PRC-02 entry and the discrimination pin.

PRC-02 runs saturation-free (the harness rerun), so the aux here does too.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from lethe.kernels.ops import fused_block_forward
from lethe.kernels.references.fused_block_forward import reference_fused_block_forward
from lethe.verifier.op_harness import FUSED_CONV_K, SCAN_N_STATE, _fused_aux


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
    *,
    fp16_readout: bool = False,
) -> torch.Tensor:
    """Cheat: conv/SiLU/norm in fp32 but the scan state carried in fp16."""
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
        c_t = cpf[:, t].unsqueeze(1)
        if fp16_readout:
            ys.append((h * c_t.half()).sum(-1).float() + dsf * z[:, t])
        else:
            ys.append((h.float() * c_t).sum(-1) + dsf * z[:, t])
    y_scan = torch.stack(ys, dim=1)
    rms = y_scan.pow(2).mean(dim=-1, keepdim=True).add(1e-5).sqrt()
    return (y_scan / rms * nwf).to(x.dtype)


def main() -> None:
    for batch, seqlen, d_model in ((2, 1024, 32), (1, 2048, 32), (1, 4096, 32)):
        worst = {"hon": 0.0, "mild": 1e9, "harsh": 1e9}
        worst_rel = {"hon": 0.0, "mild": 1e9, "harsh": 1e9}
        for seed in range(3):
            torch.manual_seed(seed)
            t32 = torch.randn(batch, seqlen, d_model)
            t16 = t32.to(torch.float16)
            aux32 = _fused_aux(
                batch,
                seqlen,
                d_model,
                SCAN_N_STATE,
                FUSED_CONV_K,
                torch.device("cpu"),
                torch.float32,
                saturate=False,
            )
            aux16 = _fused_aux(
                batch,
                seqlen,
                d_model,
                SCAN_N_STATE,
                FUSED_CONV_K,
                torch.device("cpu"),
                torch.float16,
                saturate=False,
            )
            pad32 = F.pad(t32, (0, 0, FUSED_CONV_K - 1, 0))
            pad16 = F.pad(t16, (0, 0, FUSED_CONV_K - 1, 0))
            ref = reference_fused_block_forward(pad32, *aux32, chunk_size=8)
            scale = max(1.0, ref.abs().max().item())
            hon = (
                (fused_block_forward(pad16, *aux16, chunk_size=8).float() - ref).abs().max().item()
            )
            mild = (fp16_state_fused(pad16, *aux16).float() - ref).abs().max().item()
            harsh = (
                (fp16_state_fused(pad16, *aux16, fp16_readout=True).float() - ref)
                .abs()
                .max()
                .item()
            )
            worst["hon"] = max(worst["hon"], hon)
            worst["mild"] = min(worst["mild"], mild)
            worst["harsh"] = min(worst["harsh"], harsh)
            worst_rel["hon"] = max(worst_rel["hon"], hon / scale)
            worst_rel["mild"] = min(worst_rel["mild"], mild / scale)
            worst_rel["harsh"] = min(worst_rel["harsh"], harsh / scale)
            print(
                f"shape=({batch},{seqlen},{d_model}) seed={seed} scale={scale:6.2f} "
                f"honest={hon:9.3e} mild={mild:9.3e} harsh={harsh:9.3e}"
            )
        sep_m = worst["mild"] / max(worst["hon"], 1e-12)
        sep_h = worst["harsh"] / max(worst["hon"], 1e-12)
        print(
            f"--> raw   honest<= {worst['hon']:9.3e}  mild>= {worst['mild']:9.3e} "
            f"({sep_m:5.1f}x)  harsh>= {worst['harsh']:9.3e} ({sep_h:5.1f}x)"
        )
        print(
            f"--> scale honest<= {worst_rel['hon']:9.3e}  mild>= {worst_rel['mild']:9.3e}  "
            f"harsh>= {worst_rel['harsh']:9.3e}\n"
        )


if __name__ == "__main__":
    main()
