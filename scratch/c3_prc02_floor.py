"""Measure PRC-02 floors for the MIMO backward views at the gate shape.

Honest = the eager op (fp32 compute from fp16-rounded inputs) vs the fp32
reference; cheat = autograd through a MIMO forward whose state is carried
in fp16. Prints per-view max_err and scale-normalised err over several
gate draws — the numbers that set MIMO_BWD_GATE_OVERRIDES' PRC-02 entries.
"""

from __future__ import annotations

import torch

from lethe.kernels.ops import mimo_backward
from lethe.kernels.references.mimo_backward import (
    MimoGrads,
    reference_mimo_backward,
)
from lethe.verifier.op_harness import (
    MIMO_HEADDIM,
    MIMO_N_STATE,
    MIMO_RANK,
    _mimo_bwd_aux,
)


def fp16_state_mimo_bwd(
    x: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    dt: torch.Tensor,
    alpha: torch.Tensor,
    mimo_x: torch.Tensor,
    mimo_o: torch.Tensor,
    dy: torch.Tensor,
) -> MimoGrads:
    """Cheat: discretisation in fp32 but the running state carried in fp16."""
    inputs = (x, B, C, dt, alpha, mimo_x, mimo_o)
    with torch.enable_grad():
        leaves = [t.detach().float().requires_grad_(True) for t in inputs]
        xf, bf, cf, dtf, alphaf, mxf, mof = leaves
        batch, seqlen, _, _ = xf.shape
        rank = bf.shape[2]
        mimo_x_bc = mxf.permute(1, 0, 2).unsqueeze(0).unsqueeze(0)
        x_r = xf.unsqueeze(2) * mimo_x_bc
        h = torch.zeros(batch, rank, xf.shape[2], xf.shape[3], bf.shape[4], dtype=torch.float16)
        ys = []
        mimo_o_bc = mof.permute(1, 0, 2).unsqueeze(0)
        for t in range(seqlen):
            alpha_t = alphaf[:, t].unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            dt_t = dtf[:, t].unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            b_t = bf[:, t].unsqueeze(3)
            x_r_t = x_r[:, t].unsqueeze(-1)
            h = (alpha_t.half() * h + (dt_t * b_t * x_r_t).half()).half()
            h_agg = h.float().sum(1)
            c_t = cf[:, t].unsqueeze(3)
            y_raw = (h_agg.unsqueeze(1) * c_t).sum(-1)
            ys.append((y_raw * mimo_o_bc).sum(1))
        y = torch.stack(ys, dim=1)
        grads = torch.autograd.grad(y, leaves, dy.float())
    return MimoGrads(*(g.to(x.dtype) for g in grads))


def fp16_readout_mimo_bwd(
    x: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    dt: torch.Tensor,
    alpha: torch.Tensor,
    mimo_x: torch.Tensor,
    mimo_o: torch.Tensor,
    dy: torch.Tensor,
) -> MimoGrads:
    """Harsher cheat (scan-class): fp16 state AND fp16 readout dot."""
    inputs = (x, B, C, dt, alpha, mimo_x, mimo_o)
    with torch.enable_grad():
        leaves = [t.detach().float().requires_grad_(True) for t in inputs]
        xf, bf, cf, dtf, alphaf, mxf, mof = leaves
        batch, seqlen, _, _ = xf.shape
        rank = bf.shape[2]
        mimo_x_bc = mxf.permute(1, 0, 2).unsqueeze(0).unsqueeze(0)
        x_r = xf.unsqueeze(2) * mimo_x_bc
        h = torch.zeros(batch, rank, xf.shape[2], xf.shape[3], bf.shape[4], dtype=torch.float16)
        ys = []
        mimo_o_bc = mof.permute(1, 0, 2).unsqueeze(0)
        for t in range(seqlen):
            alpha_t = alphaf[:, t].unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            dt_t = dtf[:, t].unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            b_t = bf[:, t].unsqueeze(3)
            x_r_t = x_r[:, t].unsqueeze(-1)
            h = (alpha_t.half() * h + (dt_t * b_t * x_r_t).half()).half()
            h_agg = h.sum(1)
            c_t = cf[:, t].unsqueeze(3)
            y_raw = (h_agg.unsqueeze(1) * c_t.half()).sum(-1).float()
            ys.append((y_raw * mimo_o_bc).sum(1))
        y = torch.stack(ys, dim=1)
        grads = torch.autograd.grad(y, leaves, dy.float())
    return MimoGrads(*(g.to(x.dtype) for g in grads))


def main() -> None:
    fields = MimoGrads._fields
    for batch, seqlen, d_model in ((2, 1024, 32), (1, 2048, 32), (1, 4096, 32)):
        nheads = d_model // MIMO_HEADDIM
        worst: dict[str, dict[str, float]] = {
            f: {"hon_rel": 0.0, "mild_rel": 1e9, "harsh_rel": 1e9} for f in fields
        }
        for seed in range(2):
            torch.manual_seed(seed)
            t32 = torch.randn(batch, seqlen, d_model)
            t16 = t32.to(torch.float16)
            dy32 = t32.view(batch, seqlen, nheads, MIMO_HEADDIM)
            dy16 = t16.view(batch, seqlen, nheads, MIMO_HEADDIM)
            aux32 = _mimo_bwd_aux(
                batch,
                seqlen,
                nheads,
                MIMO_HEADDIM,
                MIMO_RANK,
                MIMO_N_STATE,
                torch.device("cpu"),
                torch.float32,
            )
            aux16 = _mimo_bwd_aux(
                batch,
                seqlen,
                nheads,
                MIMO_HEADDIM,
                MIMO_RANK,
                MIMO_N_STATE,
                torch.device("cpu"),
                torch.float16,
            )
            ref = reference_mimo_backward(*aux32, dy32)
            hon = mimo_backward(*aux16, dy16)
            mild = fp16_state_mimo_bwd(*aux16, dy16)
            harsh = fp16_readout_mimo_bwd(*aux16, dy16)
            for f, r, h_, m_, x_ in zip(fields, ref, hon, mild, harsh, strict=True):
                scale = max(1.0, r.abs().max().item())
                w = worst[f]
                w["hon_rel"] = max(w["hon_rel"], (h_.float() - r).abs().max().item() / scale)
                w["mild_rel"] = min(w["mild_rel"], (m_.float() - r).abs().max().item() / scale)
                w["harsh_rel"] = min(w["harsh_rel"], (x_.float() - r).abs().max().item() / scale)
        print(f"\nshape=({batch}, {seqlen}, {d_model})  worst honest vs best cheats (rel):")
        for f, w in worst.items():
            sep_m = w["mild_rel"] / max(w["hon_rel"], 1e-12)
            sep_h = w["harsh_rel"] / max(w["hon_rel"], 1e-12)
            print(
                f"{f:13s} honest<= {w['hon_rel']:9.3e}  mild>= {w['mild_rel']:9.3e} "
                f"({sep_m:5.1f}x)  harsh>= {w['harsh_rel']:9.3e} ({sep_h:5.1f}x)"
            )


if __name__ == "__main__":
    main()
