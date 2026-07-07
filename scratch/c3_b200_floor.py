"""B200 re-measurement of the MIMO PRC-02 floors at the gate shape.

CPU calibration (c3_prc02_floor.py) set the per-view atols; this records
the Triton kernel's actual honest floor (fp16 inputs vs fp32 reference)
and the fp16-state cheat's floor on the box, per view, against those
atols. Box usage: fleet run "bash scratch/detach.sh uv run python
scratch/c3_b200_floor.py"
"""

from __future__ import annotations

import torch

from lethe.kernels.ops import mimo_backward
from lethe.kernels.references.mimo_backward import (
    MimoGrads,
    reference_mimo_backward,
)
from lethe.verifier.op_harness import (
    MIMO_BWD_GATE_OVERRIDES,
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
    inputs = (x, B, C, dt, alpha, mimo_x, mimo_o)
    with torch.enable_grad():
        leaves = [t.detach().float().requires_grad_(True) for t in inputs]
        xf, bf, cf, dtf, alphaf, mxf, mof = leaves
        batch, seqlen = xf.shape[0], xf.shape[1]
        rank = bf.shape[2]
        x_r = xf.unsqueeze(2) * mxf.permute(1, 0, 2).unsqueeze(0).unsqueeze(0)
        h = torch.zeros(
            batch, rank, xf.shape[2], xf.shape[3], bf.shape[4],
            dtype=torch.float16, device=xf.device,
        )
        mimo_o_bc = mof.permute(1, 0, 2).unsqueeze(0)
        ys = []
        for t in range(seqlen):
            alpha_t = alphaf[:, t].unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            dt_t = dtf[:, t].unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            b_t = bf[:, t].unsqueeze(3)
            x_r_t = x_r[:, t].unsqueeze(-1)
            h = (alpha_t.half() * h + (dt_t * b_t * x_r_t).half()).half()
            h_agg = h.float().sum(1)
            y_raw = (h_agg.unsqueeze(1) * cf[:, t].unsqueeze(3)).sum(-1)
            ys.append((y_raw * mimo_o_bc).sum(1))
        y = torch.stack(ys, dim=1)
        grads = torch.autograd.grad(y, leaves, dy.float())
    return MimoGrads(*(g.to(x.dtype) for g in grads))


def main() -> None:
    dev = torch.device("cuda")
    batch, seqlen, d_model = 1, 4096, 32
    nheads = d_model // MIMO_HEADDIM
    fields = MimoGrads._fields
    atols = {
        f: MIMO_BWD_GATE_OVERRIDES[f]["gate_prc_02_mixed_precision_accumulation"]["atol"]
        for f in fields
    }
    worst = {f: {"hon": 0.0, "che": 1e9} for f in fields}
    for seed in range(3):
        torch.manual_seed(seed)
        t32 = torch.randn(batch, seqlen, d_model, device=dev)
        t16 = t32.to(torch.float16)
        dy32 = t32.view(batch, seqlen, nheads, MIMO_HEADDIM)
        dy16 = t16.view(batch, seqlen, nheads, MIMO_HEADDIM)
        aux32 = _mimo_bwd_aux(
            batch, seqlen, nheads, MIMO_HEADDIM, MIMO_RANK, MIMO_N_STATE, dev, torch.float32
        )
        aux16 = _mimo_bwd_aux(
            batch, seqlen, nheads, MIMO_HEADDIM, MIMO_RANK, MIMO_N_STATE, dev, torch.float16
        )
        ref = reference_mimo_backward(*aux32, dy32)
        hon = mimo_backward(*aux16, dy16)  # Triton kernel on CUDA fp16
        che = fp16_state_mimo_bwd(*aux16, dy16)
        for f, r, h_, c_ in zip(fields, ref, hon, che, strict=True):
            scale = max(1.0, r.abs().max().item())
            he = (h_.float() - r).abs().max().item() / scale
            ce = (c_.float() - r).abs().max().item() / scale
            worst[f]["hon"] = max(worst[f]["hon"], he)
            worst[f]["che"] = min(worst[f]["che"], ce)
            print(f"seed={seed} {f:13s} kernel={he:9.3e}  cheat={ce:9.3e}  atol={atols[f]:.1e}")
    print("\nB200 verdict per view (kernel worst vs cheat best vs atol):")
    for f, w in worst.items():
        margin_h = atols[f] / max(w["hon"], 1e-12)
        margin_c = w["che"] / atols[f]
        print(
            f"{f:13s} kernel<= {w['hon']:9.3e} ({margin_h:4.1f}x under atol)  "
            f"cheat>= {w['che']:9.3e} ({margin_c:4.1f}x over atol)"
        )


if __name__ == "__main__":
    main()
