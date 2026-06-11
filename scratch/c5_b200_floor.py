"""B200 re-measurement of the fused-block PRC-02 floors at the gate shape.

CPU calibration (c5_prc02_floor.py) measured the eager op's honest floor;
this records the Triton kernels' actual honest floor (fp16 inputs vs fp32
reference) and both fp16-state cheats on the box, as fractions of output
scale against the gate's scale-aware unit atol (5e-3 at the (1, 4096, 32)
gate shape). PRC-02 runs saturation-free, so the aux here does too.
Box usage:
fleet run "bash scratch/detach.sh uv run python scratch/c5_b200_floor.py"
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from flash_mamba_rl.kernels.ops import fused_block_forward
from flash_mamba_rl.kernels.references.fused_block_forward import reference_fused_block_forward
from flash_mamba_rl.verifier.op_harness import FUSED_CONV_K, SCAN_N_STATE, _fused_aux

PRC02_UNIT_ATOL = 5e-3


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
    h = torch.zeros(batch, d_model, af.shape[1], dtype=torch.float16, device=xf.device)
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
    dev = torch.device("cuda")
    batch, seqlen, d_model = 1, 4096, 32
    worst = {"hon": 0.0, "mild": 1e9, "harsh": 1e9}
    worst_abs = {"hon": 0.0, "mild": 1e9, "harsh": 1e9}
    for seed in range(3):
        torch.manual_seed(seed)
        t32 = torch.randn(batch, seqlen, d_model, device=dev)
        t16 = t32.to(torch.float16)
        aux32 = _fused_aux(
            batch, seqlen, d_model, SCAN_N_STATE, FUSED_CONV_K, dev, torch.float32, saturate=False
        )
        aux16 = _fused_aux(
            batch, seqlen, d_model, SCAN_N_STATE, FUSED_CONV_K, dev, torch.float16, saturate=False
        )
        pad32 = F.pad(t32, (0, 0, FUSED_CONV_K - 1, 0))
        pad16 = F.pad(t16, (0, 0, FUSED_CONV_K - 1, 0))
        ref = reference_fused_block_forward(pad32, *aux32, chunk_size=8)
        scale = max(1.0, ref.abs().max().item())
        hon_abs = (
            (fused_block_forward(pad16, *aux16, chunk_size=8).float() - ref).abs().max().item()
        )
        mild_abs = (fp16_state_fused(pad16, *aux16).float() - ref).abs().max().item()
        harsh_abs = (
            (fp16_state_fused(pad16, *aux16, fp16_readout=True).float() - ref).abs().max().item()
        )
        worst["hon"] = max(worst["hon"], hon_abs / scale)
        worst["mild"] = min(worst["mild"], mild_abs / scale)
        worst["harsh"] = min(worst["harsh"], harsh_abs / scale)
        worst_abs["hon"] = max(worst_abs["hon"], hon_abs)
        worst_abs["mild"] = min(worst_abs["mild"], mild_abs)
        worst_abs["harsh"] = min(worst_abs["harsh"], harsh_abs)
        print(
            f"seed={seed} scale={scale:7.3f} "
            f"kernel={hon_abs / scale:9.3e} mild={mild_abs / scale:9.3e} "
            f"harsh={harsh_abs / scale:9.3e} (of scale)"
        )
    sep_m = worst["mild"] / max(worst["hon"], 1e-12)
    sep_h = worst["harsh"] / max(worst["hon"], 1e-12)
    print(
        f"\nB200 scale-rel: kernel<= {worst['hon']:9.3e}  "
        f"mild>= {worst['mild']:9.3e} ({sep_m:5.1f}x)  "
        f"harsh>= {worst['harsh']:9.3e} ({sep_h:5.1f}x)"
    )
    margin_h = PRC02_UNIT_ATOL / max(worst["hon"], 1e-12)
    margin_m = worst["mild"] / PRC02_UNIT_ATOL
    margin_hh = worst["harsh"] / PRC02_UNIT_ATOL
    print(
        f"vs scale-aware unit atol {PRC02_UNIT_ATOL:.0e}: "
        f"kernel {margin_h:4.1f}x under  mild {margin_m:4.1f}x over  "
        f"harsh {margin_hh:4.1f}x over"
        f"  (abs for reference: kernel<= {worst_abs['hon']:9.3e} "
        f"mild>= {worst_abs['mild']:9.3e})"
    )


if __name__ == "__main__":
    main()
