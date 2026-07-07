"""Desk repro for the native-training NaN (burst-2 tiny-train step-3 fault).

Hypothesis: the chunkwise assembly computes ``decay_rel = exp2(g2_i - g2_s)`` UNMASKED;
the upper-triangle exponents are positive and overflow fp32 to inf once the per-chunk
log2-decay span exceeds ~128 (mean per-token g below ~-1.39 at C=64). The strictly-lower
``d_m``/``da_qk`` factors are exact zeros there, so the einsums produce ``0 * inf = NaN``.
The token-serial eager path never forms these matrices -> eager trains, native NaNs.

Probe: build a drifted-regime input (g mean ~-1.6/token), run the SAME operands through
  (a) the cw assembly, default autograd stage-B  (the tiny-train native path)
  (b) the cw assembly, stage_b_closed=True
  (c) the token-serial eager fallback (gdn2_backward with native unavailable)
and localize the first non-finite site per component (K#2 ref, stage-B, forward).
Pure CPU fp32; the pure-torch kernel refs stand in for the box kernels (identical glue).
"""

from __future__ import annotations

import torch

from lethe.kernels.cute.gdn2_assemble import (
    _stage_b_vjp_cw,
    _stage_b_vjp_cw_closed,
    _to_chunks,
    assembled_channelwise_gdn2_backward,
    k1_reverse_state_cw_ref,
    k2_wy_vjp_cw_ref,
)
from lethe.kernels.ops.gdn_backward import gdn2_backward
from lethe.kernels.references.gdn2_chunkwise_cw import chunkwise_forward_cw

torch.manual_seed(0)

B, L, H, DK, DV = 2, 128, 2, 32, 32
MEAN_G = -1.6  # per-token log-decay; span_log2 ~ 63*1.6*1.4427 ~ 145 > 128 (fp32 exp2 cliff)

q = torch.randn(B, L, H, DK)
k = torch.randn(B, L, H, DK)
v = torch.randn(B, L, H, DV)
g = MEAN_G - 0.5 * torch.rand(B, L, H, DK)
b = torch.sigmoid(torch.randn(B, L, H, DK))
w = torch.sigmoid(torch.randn(B, L, H, DV))
do = torch.randn(B, L, H, DV)


def finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all())


def report(tag: str, grads: object) -> None:
    fields = [f for f in dir(grads) if f.startswith(("grad_", "d")) and not f.startswith("__")]
    bad = []
    for f in fields:
        t = getattr(grads, f, None)
        if isinstance(t, torch.Tensor) and not finite(t):
            bad.append(f)
    print(f"[{tag}] non-finite grads: {bad if bad else 'NONE (all finite)'}")


# --- exponent inventory -------------------------------------------------------
g2 = torch.cumsum(_to_chunks(g, 64), dim=-2) * (1.0 / torch.log(torch.tensor(2.0)))
diff = g2[..., :, None, :] - g2[..., None, :, :]
print(f"max positive decay_rel exponent (log2): {diff.max().item():.1f}  (fp32 inf above ~128)")
print(f"decay_rel has inf: {bool(torch.isinf(torch.exp2(diff)).any())}")

# --- (c) eager token-serial ---------------------------------------------------
eager = gdn2_backward(q.cuda() if False else q, k, v, g, b, w, do)
report("eager token-serial", eager)

# --- forward + component-level localization -----------------------------------
fwd = chunkwise_forward_cw(q, k, v, g, b, w, chunk_len=64, use_qk_l2norm=True)
print(f"forward o finite: {finite(fwd.o)}   T finite: {finite(fwd.T)}   "
      f"wy finite: {finite(fwd.wy)}   max|T|: {fwd.T.abs().max().item():.3e}   "
      f"max|wy|: {fwd.wy.abs().max().item():.3e}")

do_c = _to_chunks(do, 64)
dv_local = fwd.A_qk.transpose(-1, -2) @ do_c
dht = torch.zeros_like(fwd.h_list[0])

dq_b, dk_b, dg_b, dwy, _du, _dh0 = _stage_b_vjp_cw(fwd, do, create_graph=False)
print(f"stage-B autograd: dq_b {finite(dq_b)}  dk_b {finite(dk_b)}  dg_b {finite(dg_b)}  "
      f"dwy {finite(dwy)}")

dh_k1, dv2, dh0 = k1_reverse_state_cw_ref(
    fwd.q.detach(), fwd.k.detach(), fwd.wy.detach(), fwd.g2.detach(), fwd.g_last.detach(),
    do_c, dv_local.detach(), dht,
)
print(f"K#1 ref: dh {finite(dh_k1)}  dv2 {finite(dv2)}  dh0 {finite(dh0)}  "
      f"max|dv2|: {dv2.abs().max().item():.3e}  max|b_dh|: {dh_k1.abs().max().item():.3e}")

cq, ck, cg, cwy = _stage_b_vjp_cw_closed(fwd, do, dh_k1, dv2)
print(f"stage-B closed: dq {finite(cq)}  dk {finite(ck)}  dg {finite(cg)}  dwy {finite(cwy)}")

dk2, dvf, dbf, dwf, dg2f = k2_wy_vjp_cw_ref(
    fwd.k.detach(), fwd.v.detach(), fwd.b.detach(), fwd.w_gate.detach(), fwd.g2.detach(),
    fwd.T.detach(), dwy.detach() if finite(dwy) else torch.zeros_like(dwy), dv2.detach(),
)
print(f"K#2 ref: dk2 {finite(dk2)}  dv {finite(dvf)}  db {finite(dbf)}  dw {finite(dwf)}  "
      f"dg2 {finite(dg2f)}")

# --- (a)/(b) full assemblies ---------------------------------------------------
grads_a = assembled_channelwise_gdn2_backward(q, k, v, g, b, w, do)
report("assembly default (autograd stage-B)", grads_a)
grads_b = assembled_channelwise_gdn2_backward(q, k, v, g, b, w, do, stage_b_closed=True)
report("assembly stage_b_closed", grads_b)

# --- f16 operand-range inventory (the second suspected cliff, box kernels only) --
print("\nf16 operand ranges (65504 cliff) in this regime:")
for name, t in (("T", fwd.T), ("wy", fwd.wy), ("b_dh(dh)", dh_k1), ("dv2", dv2),
                ("dwy", dwy), ("u", fwd.u), ("h", fwd.h)):
    print(f"  max|{name}| = {t.abs().max().item():.3e}")
