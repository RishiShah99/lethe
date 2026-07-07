"""M4 landing gate: do the fixed-seed contract gates still discriminate the
MIMO PRC-02 cheat on B200?

Seeding run_all_gates (contracts._GATE_RNG_SEED) pins each gate to one draw.
The thinnest discriminated view is grad_dt PRC-02 (~1.2x honest margin / ~1.3x
cheat margin on B200). This runs the honest Triton kernel and the fp16-state-
carry cheat through the EXACT seeded gate path on CUDA and asserts: the honest
kernel passes every gate on every view, and the cheat is rejected on grad_dt
(via PRC-02). A fixed seed that lands grad_dt on the wrong side trips this.

Box: fleet run "bash scratch/detach.sh uv run python scratch/m4_seed_validate.py"
"""

from __future__ import annotations

import torch

from lethe.kernels.ops import mimo_backward
from lethe.kernels.ops.mimo_backward import triton_mimo_bwd_resource_meta
from lethe.kernels.references.mimo_backward import MimoGrads
from lethe.verifier.op_harness import (
    MIMO_HEADDIM,
    MIMO_N_STATE,
    MIMO_RANK,
    _mimo_bwd_aux,
    verify_mimo_bwd_op_all_grads,
)

_PRC02 = "gate_prc_02_mixed_precision_accumulation"


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
    """Carry the recurrence state in fp16 — the PRC-02 violation (copied from
    scratch/c3_b200_floor.py so this probe is self-contained)."""
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


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required")
        return 2
    dev = torch.device("cuda")
    nheads = 4
    aux = _mimo_bwd_aux(1, 8, nheads, MIMO_HEADDIM, MIMO_RANK, MIMO_N_STATE, dev, torch.float32)
    dy = torch.randn(1, 8, nheads, MIMO_HEADDIM, device=dev)
    mimo_backward(*aux, dy)  # warm the kernel cache for resource_meta
    meta = triton_mimo_bwd_resource_meta()

    honest = verify_mimo_bwd_op_all_grads(mimo_backward, device="cuda", resource_meta=meta)
    cheat = verify_mimo_bwd_op_all_grads(fp16_state_mimo_bwd, device="cuda")

    ok = True
    print("=== honest kernel (must pass every gate on every view) ===")
    for view, results in honest.items():
        failed = {n: r.reason for n, r in results.items() if not r.passed}
        if failed:
            ok = False
            print(f"  {view}: FAILED {failed}")
        else:
            print(f"  {view}: all pass")

    print("=== fp16-state cheat (must be rejected; PRC-02 the documented discriminator) ===")
    cheat_prc_fail: list[str] = []
    for view, results in cheat.items():
        prc = results.get(_PRC02)
        prc_failed = prc is not None and not prc.passed
        if prc_failed:
            cheat_prc_fail.append(view)
        other = [n for n, r in results.items() if not r.passed and n != _PRC02]
        print(f"  {view}: PRC-02={'FAIL' if prc_failed else 'pass'} other_fails={other}")
    if "grad_dt" not in cheat_prc_fail:
        ok = False
        print("  !! cheat PASSED PRC-02 on grad_dt — the fixed seed lands the thin view wrong")

    print(f"M4_SEED_OK={ok} cheat_prc_fail_views={cheat_prc_fail}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
