"""Desk proof of the GDN-2 backward ASSEMBLY (no kernel, no box).

The integration (#19) wires ``native_gdn2_backward`` to assemble the six grads
from the two hard kernels (K#1 reverse-state scan, K#2 WY-VJP) plus supporting
torch stages. This script proves the assembly *design* is bit-correct in fp64
before any of it is productionised, by routing K#1/K#2 to pure-torch reference
paths and comparing the assembled grads against the token-serial oracle.

Design = a two-stage VJP splice over the canonical chunkwise forward:

  * Stage A (B1 map): (k, v, beta, g) -> (u, w).  VJP == K#2 (run_k2_ref).
  * Stage B (everything else): (q, k, g, u, w, h0) -> o, with u/w CUT as leaves.
    VJP gives dq, dk_B, dg_B (supporting), dw (-> K#2), and du/dh0 (== K#1).

K#1 owns du(=dv2) + dh0 (the reverse inter-chunk state recurrence); K#2 owns the
B1-map VJP. Final grads: dq=dq_B; dv=K#2.dv; dk=dk_B+K#2.dk2; dg=dg_B+K#2.dg2;
db=K#2.db; then the B8 L2-norm VJP maps dq/dk from normed back to raw q/k.

Scalar regime (Phase 2): g scalar per token, b=w=beta. The oracle is fed
channel-constant g and beta-broadcast b/w, so its channel-wise grad_g/grad_b/
grad_w reduce by sum-over-channel to the scalar assembly's dg/db.

Run: PYTHONPATH=src uv run --no-sync python scratch/gdn2_assemble_desk_check.py
"""

from __future__ import annotations

import torch

# pure-torch K#2 (the validated WY-VJP spec)
from scratch.gdn2_bwd_wy import run_k2_ref

# pure-torch K#1 (the increment-B reverse loop, kernel statement order)
from scratch.k1_incB_desk_check import reverse_loop

from flash_mamba_rl.kernels.references.gdn2_chunkwise import (
    RCP_LN2,
    ChunkwiseForward,
    _from_chunks,
    _l2norm,
    chunkwise_forward,
)
from flash_mamba_rl.kernels.references.gdn_backward import reference_gdn2_backward


def _stage_b_vjp(
    fwd: ChunkwiseForward, do: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """VJP of Stage B (u, w, h0 cut as leaves) -> dq, dk_B, dg_B, dw, du, dh0.

    Rebuilds the chunkwise forward's recurrence + read with u/w supplied as fresh
    leaves (so k/g grads here exclude the B1 map, which K#2 owns). Leaves are the
    scaled-normed q and normed k the kernel sees; g is the raw per-token scalar.
    """
    dtype, dev = fwd.q.dtype, fwd.q.device
    b, h, nt, c, d_k = fwd.q.shape

    q_sn = fwd.q.detach().clone().requires_grad_(True)  # scaled, normed
    k_n = fwd.k.detach().clone().requires_grad_(True)  # normed
    u_lf = fwd.u.detach().clone().requires_grad_(True)
    w_lf = fwd.w.detach().clone().requires_grad_(True)
    h0_lf = fwd.h_list[0].detach().clone().requires_grad_(True)

    # raw per-token g recovered from the within-chunk log2 cumsum (= the K#2 space).
    g_tok = torch.diff(
        fwd.g2 / RCP_LN2, dim=-1, prepend=torch.zeros(b, h, nt, 1, dtype=dtype, device=dev)
    )
    g_lf = g_tok.detach().clone().requires_grad_(True)

    g2 = torch.cumsum(g_lf, dim=-1) * RCP_LN2
    gamma = torch.exp2(g2)
    g_last = g2[..., -1]
    lower_incl = torch.tril(torch.ones(c, c, dtype=torch.bool, device=dev), 0)

    do_c = do.transpose(1, 2).reshape(b, h, nt, c, do.shape[-1]) if do.dim() == 4 else do

    o_list = []
    hh = h0_lf
    for ci in range(nt):
        gamma_c, g2_c, glast_c = gamma[:, :, ci], g2[:, :, ci], g_last[:, :, ci]
        ratio = gamma_c[..., :, None] / gamma_c[..., None, :]
        a_qk = (q_sn[:, :, ci] @ k_n[:, :, ci].transpose(-1, -2)) * ratio
        a_qk = torch.where(lower_incl, a_qk, torch.zeros_like(a_qk))
        v_new = u_lf[:, :, ci] - w_lf[:, :, ci] @ hh
        qg = q_sn[:, :, ci] * gamma_c[..., None]
        o_list.append(qg @ hh + a_qk @ v_new)
        decay_end = torch.exp2(glast_c[..., None] - g2_c)
        k_dec = k_n[:, :, ci] * decay_end[..., None]
        hh = torch.exp2(glast_c)[..., None, None] * hh + k_dec.transpose(-1, -2) @ v_new

    o = torch.stack(o_list, dim=2)
    dq_b, dk_b, dg_b, dw, du, dh0 = torch.autograd.grad(
        o, (q_sn, k_n, g_lf, w_lf, u_lf, h0_lf), do_c
    )
    return dq_b, dk_b, dg_b, dw, du, dh0


def assemble(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    *,
    chunk_len: int,
    scale: float | None,
) -> dict[str, torch.Tensor]:
    """Full assembly with K#1/K#2 on their pure-torch reference paths."""
    fwd = chunkwise_forward(q, k, v, g, beta, chunk_len=chunk_len, scale=scale, use_qk_l2norm=True)
    dq_b, dk_b, dg_b, dw, _du_b, _dh0_b = _stage_b_vjp(fwd, do)

    # --- K#1: dv2 (= du fed to K#2) + dh0, via the reverse-loop reference ---
    do_c = do.transpose(1, 2).reshape(*fwd.q.shape[:3], fwd.q.shape[3], do.shape[-1])
    dv_local = fwd.A_qk.transpose(-1, -2) @ do_c
    k1_in = {
        "q": fwd.q,
        "k": fwd.k,
        "w": fwd.w,
        "g2": fwd.g2,
        "g_last": fwd.g_last,
        "do": do_c,
        "dv_local": dv_local,
        "dht": torch.zeros_like(fwd.h_list[0]),
    }
    k1 = reverse_loop(k1_in)
    dv2, dh0 = k1["dv2"], k1["dh0"]

    # --- K#2: B1-map VJP from (du=dv2, dw) ---
    dk2, dv, db, dg2 = run_k2_ref(fwd.k, fwd.v, fwd.beta, fwd.g2, fwd.T, dw, dv2)
    db_erase, dw_write = _k2_beta_split(fwd, dw, dv2)

    # --- combine in normed space, then B8 L2-norm VJP back to raw q/k ---
    dq_n = _from_chunks(dq_b)
    dk_n = _from_chunks(dk_b + dk2)
    dg = _from_chunks((dg_b + dg2).unsqueeze(-1)).squeeze(-1)
    dv_f = _from_chunks(dv)
    dbeta = _from_chunks(db.unsqueeze(-1)).squeeze(-1)

    s = q.shape[-1] ** -0.5 if scale is None else scale
    q_lf = q.detach().clone().requires_grad_(True)
    k_lf = k.detach().clone().requires_grad_(True)
    q_sn = _l2norm(q_lf) * s
    k_nn = _l2norm(k_lf)
    dq, dk = torch.autograd.grad((q_sn, k_nn), (q_lf, k_lf), (dq_n, dk_n))

    db_e = _from_chunks(db_erase.unsqueeze(-1)).squeeze(-1)
    dw_w = _from_chunks(dw_write.unsqueeze(-1)).squeeze(-1)
    return {
        "dq": dq,
        "dk": dk,
        "dv": dv_f,
        "dg": dg,
        "db": dbeta,
        "db_erase": db_e,
        "dw_write": dw_w,
        "dh0": dh0,
    }


def _k2_beta_split(
    fwd: ChunkwiseForward, dw: torch.Tensor, du: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the lumped beta grad into (erase=key-side, write=value-side) parts.

    In b=w=beta the K#2 db lumps both gate paths. The v-side (bv=beta*v) is the
    write gate w; the k-side (bgk=beta*gamma*k plus the M=beta*ratio*kk term) is
    the erase gate b. Verified here against the oracle's separate grad_b/grad_w.
    """
    b, h, nt, c, _ = fwd.k.shape

    def _f(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(b * h * nt, *x.shape[3:])

    kf, vf, betaf, g2f, tf = _f(fwd.k), _f(fwd.v), _f(fwd.beta), _f(fwd.g2), _f(fwd.T)
    dwf, duf = _f(dw), _f(du)
    sl = torch.tril(torch.ones(c, c, dtype=torch.bool, device=kf.device), -1)
    gamma = torch.exp2(g2f)
    bv = betaf[..., None] * vf
    bgk = (betaf * gamma)[..., None] * kf
    kk = kf @ kf.transpose(-1, -2)
    ratio = gamma[..., :, None] / gamma[..., None, :]
    tt = tf.transpose(-1, -2)
    dbv = tt @ duf
    dbgk = tt @ dwf
    d_t = duf @ bv.transpose(-1, -2) + dwf @ bgk.transpose(-1, -2)
    d_m = torch.where(sl, -(tt @ d_t @ tt), torch.zeros_like(d_t))
    r = (dbv * vf).sum(-1)  # write (v-side)
    p = (dbgk * kf).sum(-1)
    db_erase = gamma * p + (d_m * (ratio * kk)).sum(-1)  # erase (k-side)
    dw_write = r
    return (db_erase.reshape(b, h, nt, c), dw_write.reshape(b, h, nt, c))


def main() -> None:
    torch.manual_seed(0)
    b, t, h, d = 2, 128, 3, 16
    chunk_len = 64
    dt = torch.float64

    q = torch.randn(b, t, h, d, dtype=dt)
    k = torch.randn(b, t, h, d, dtype=dt)
    v = torch.randn(b, t, h, d, dtype=dt)
    g_scalar = -torch.rand(b, t, h, dtype=dt) * 0.1
    beta = torch.rand(b, t, h, dtype=dt) * 0.8 + 0.1
    do = torch.randn(b, t, h, d, dtype=dt)

    got = assemble(q, k, v, g_scalar, beta, do, chunk_len=chunk_len, scale=None)

    # oracle in the reduction regime: channel-constant g, beta-broadcast b/w.
    g_chan = g_scalar.unsqueeze(-1).expand(b, t, h, d).contiguous()
    b_gate = beta.unsqueeze(-1).expand(b, t, h, d).contiguous()
    w_gate = beta.unsqueeze(-1).expand(b, t, h, d).contiguous()
    orc = reference_gdn2_backward(q, k, v, g_chan, b_gate, w_gate, do)

    checks = {
        "dq": (got["dq"], orc.grad_q),
        "dk": (got["dk"], orc.grad_k),
        "dv": (got["dv"], orc.grad_v),
        "dg": (got["dg"], orc.grad_g.sum(-1)),
        "db": (got["db"], orc.grad_b.sum(-1) + orc.grad_w.sum(-1)),
        "db_erase": (got["db_erase"], orc.grad_b.sum(-1)),
        "dw_write": (got["dw_write"], orc.grad_w.sum(-1)),
    }
    ok = True
    for name, (a, e) in checks.items():
        diff = (a - e).abs()
        scale = e.abs().max().clamp_min(1e-12)
        rel = (diff.max() / scale).item()
        passed = rel < 1e-9
        ok = ok and passed
        print(
            f"  {name:3s} max_abs={diff.max().item():.3e}  scale_rel={rel:.3e}  {'OK' if passed else 'FAIL'}"
        )
    print(f"\nassembly_correct={ok}")


if __name__ == "__main__":
    main()
