"""Measure PRC-02 floors for the rope-scan view at candidate gate shapes.

Honest = the eager op (fp32 compute from fp16-rounded inputs) vs the fp32
reference; cheats = the same scan with the state carried in fp16 (mild)
and additionally an fp16 readout dot (harsh). Prints per-shape max_err and
scale-normalised err over several gate draws — the numbers behind
ROPE_GATE_OVERRIDES' PRC-02 entry and the discrimination pin.
"""

from __future__ import annotations

import math

import torch

from lethe.kernels.ops import complex_scan_rope
from lethe.kernels.references.complex_scan_rope import (
    _apply_rope_rotation,
    reference_complex_scan_rope,
)
from lethe.verifier.op_harness import (
    ROPE_HEADDIM,
    ROPE_N_STATE,
    ROPE_NUM_ANGLES,
    _rope_aux,
)


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
    h = torch.zeros(batch, nheads, headdim, bf.shape[-1], dtype=torch.float16)
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
    for batch, seqlen, d_model in ((2, 1024, 32), (1, 2048, 32), (1, 4096, 32)):
        nheads = d_model // ROPE_HEADDIM
        worst = {"hon": 0.0, "mild": 1e9, "harsh": 1e9}
        for seed in range(3):
            torch.manual_seed(seed)
            t32 = torch.randn(batch, seqlen, d_model)
            t16 = t32.to(torch.float16)
            x32 = t32.view(batch, seqlen, nheads, ROPE_HEADDIM)
            x16 = t16.view(batch, seqlen, nheads, ROPE_HEADDIM)
            aux32 = _rope_aux(
                batch,
                seqlen,
                nheads,
                ROPE_N_STATE,
                ROPE_NUM_ANGLES,
                torch.device("cpu"),
                torch.float32,
            )
            aux16 = _rope_aux(
                batch,
                seqlen,
                nheads,
                ROPE_N_STATE,
                ROPE_NUM_ANGLES,
                torch.device("cpu"),
                torch.float16,
            )
            ref = reference_complex_scan_rope(x32, *aux32)
            scale = max(1.0, ref.abs().max().item())
            hon = (complex_scan_rope(x16, *aux16).float() - ref).abs().max().item() / scale
            mild = (fp16_state_rope(x16, *aux16).float() - ref).abs().max().item() / scale
            harsh = (
                fp16_state_rope(x16, *aux16, fp16_readout=True).float() - ref
            ).abs().max().item() / scale
            worst["hon"] = max(worst["hon"], hon)
            worst["mild"] = min(worst["mild"], mild)
            worst["harsh"] = min(worst["harsh"], harsh)
            print(
                f"shape=({batch},{seqlen},{d_model}) seed={seed} "
                f"honest={hon:9.3e} mild={mild:9.3e} harsh={harsh:9.3e}"
            )
        sep_m = worst["mild"] / max(worst["hon"], 1e-12)
        sep_h = worst["harsh"] / max(worst["hon"], 1e-12)
        print(
            f"--> worst honest<= {worst['hon']:9.3e}  mild>= {worst['mild']:9.3e} "
            f"({sep_m:5.1f}x)  harsh>= {worst['harsh']:9.3e} ({sep_h:5.1f}x)\n"
        )


if __name__ == "__main__":
    main()
