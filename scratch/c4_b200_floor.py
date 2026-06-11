"""B200 re-measurement of the rope-scan PRC-02 floors at the gate shape.

CPU calibration (c4_prc02_floor.py) measured the eager op's honest floor;
this records the Triton kernel's actual honest floor (fp16 inputs vs fp32
reference) and both fp16-state cheats on the box, against the gate's flat
default atol (2e-2 at the (2, 1024, 32) gate shape) and as fractions of
output scale for comparison with the CPU corridor. Box usage:
fleet run "bash scratch/detach.sh uv run python scratch/c4_b200_floor.py"
"""

from __future__ import annotations

import math

import torch

from flash_mamba_rl.kernels.ops import complex_scan_rope
from flash_mamba_rl.kernels.references.complex_scan_rope import (
    _apply_rope_rotation,
    reference_complex_scan_rope,
)
from flash_mamba_rl.verifier.op_harness import (
    ROPE_HEADDIM,
    ROPE_N_STATE,
    ROPE_NUM_ANGLES,
    _rope_aux,
)

PRC02_DEFAULT_ATOL = 2e-2


def _rotated_bc(
    B: torch.Tensor, C: torch.Tensor, dt: torch.Tensor, angle_proj: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    delta_angle = torch.tanh(angle_proj) * dt.unsqueeze(-1) * math.pi
    theta = torch.remainder(torch.cumsum(delta_angle, dim=1), 2.0 * math.pi)
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    bl = B.shape[0] * B.shape[1] * B.shape[2]
    n_state = B.shape[-1]
    num_rope = angle_proj.shape[-1]
    b_rot = _apply_rope_rotation(
        B.reshape(bl, n_state), cos_t.reshape(bl, num_rope), sin_t.reshape(bl, num_rope)
    ).reshape(B.shape)
    c_rot = _apply_rope_rotation(
        C.reshape(bl, n_state), cos_t.reshape(bl, num_rope), sin_t.reshape(bl, num_rope)
    ).reshape(C.shape)
    return b_rot, c_rot


def fp16_state_rope(
    x: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    angle_proj: torch.Tensor,
    *,
    fp16_readout: bool = False,
) -> torch.Tensor:
    """Cheat: rotation/discretisation in fp32 but the running state in fp16."""
    xf, bf, cf, dtf, af, angf = (t.float() for t in (x, B, C, dt, A, angle_proj))
    b_rot, c_rot = _rotated_bc(bf, cf, dtf, angf)
    alpha = torch.exp(dtf * af)
    batch, seqlen, nheads, headdim = xf.shape
    h = torch.zeros(batch, nheads, headdim, bf.shape[-1], dtype=torch.float16, device=xf.device)
    ys = []
    for t in range(seqlen):
        alpha_t = alpha[:, t].unsqueeze(-1).unsqueeze(-1)
        bu = (dtf[:, t].unsqueeze(-1) * b_rot[:, t]).unsqueeze(2) * xf[:, t].unsqueeze(-1)
        h = (alpha_t.half() * h + bu.half()).half()
        c_t = c_rot[:, t].unsqueeze(2)
        if fp16_readout:
            ys.append((h * c_t.half()).sum(-1).float())
        else:
            ys.append((h.float() * c_t).sum(-1))
    return torch.stack(ys, dim=1).to(x.dtype)


def main() -> None:
    dev = torch.device("cuda")
    batch, seqlen, d_model = 2, 1024, 32
    nheads = d_model // ROPE_HEADDIM
    worst = {"hon": 0.0, "mild": 1e9, "harsh": 1e9}
    worst_abs = {"hon": 0.0, "mild": 1e9, "harsh": 1e9}
    for seed in range(3):
        torch.manual_seed(seed)
        t32 = torch.randn(batch, seqlen, d_model, device=dev)
        t16 = t32.to(torch.float16)
        x32 = t32.view(batch, seqlen, nheads, ROPE_HEADDIM)
        x16 = t16.view(batch, seqlen, nheads, ROPE_HEADDIM)
        aux32 = _rope_aux(batch, seqlen, nheads, ROPE_N_STATE, ROPE_NUM_ANGLES, dev, torch.float32)
        aux16 = _rope_aux(batch, seqlen, nheads, ROPE_N_STATE, ROPE_NUM_ANGLES, dev, torch.float16)
        ref = reference_complex_scan_rope(x32, *aux32)
        scale = max(1.0, ref.abs().max().item())
        hon_abs = (complex_scan_rope(x16, *aux16).float() - ref).abs().max().item()
        mild_abs = (fp16_state_rope(x16, *aux16).float() - ref).abs().max().item()
        harsh_abs = (
            (fp16_state_rope(x16, *aux16, fp16_readout=True).float() - ref).abs().max().item()
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
    margin_h = PRC02_DEFAULT_ATOL / max(worst_abs["hon"], 1e-12)
    margin_m = worst_abs["mild"] / PRC02_DEFAULT_ATOL
    margin_hh = worst_abs["harsh"] / PRC02_DEFAULT_ATOL
    print(
        f"vs flat atol {PRC02_DEFAULT_ATOL:.0e}: kernel<= {worst_abs['hon']:9.3e} "
        f"({margin_h:4.1f}x under)  mild>= {worst_abs['mild']:9.3e} ({margin_m:4.1f}x over)  "
        f"harsh>= {worst_abs['harsh']:9.3e} ({margin_hh:4.1f}x over)"
    )


if __name__ == "__main__":
    main()
