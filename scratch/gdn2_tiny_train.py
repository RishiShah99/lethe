"""Tiny GDN-2 training run — purchases the "trains the family" claim.

A 2-layer toy sequence model (embed -> [mixer + MLP] x2 -> head) trained on a
deterministic synthetic delayed-copy task (delay=1: predict the previous token).
Arms selectable via ``--arm``:

  native   — (default) native dispatch; on B200 the tcgen05 assembly runs via gdn2_backward
  fla      — fla.ops.gdn2.chunk_gdn2 if importable (box-only); skips gracefully otherwise
  eager    — monkeypatches is_available to False, forcing the oracle-faithful eager path
  assembly — routes backward through assembled_channelwise_gdn2_backward with the
             pure-torch kernel refs (CPU-runnable): the native path's exact glue minus
             the tcgen05 GEMMs. The desk gate for the drifted-regime NaN fix.
  gla/la/ssd/kda — FAMILY training arms: the mixer emits that family's gate set and the
             backward routes through native_{gla,la,ssd,kda}_backward (forward = the
             family's token-serial oracle). ``backward_dispatch`` in the result counts
             native vs fallback calls — the family gate additionally requires
             fallback == 0 on CUDA, so a silent eager fallback cannot fabricate the
             "trains the family" purchase.

Envelope (enforced for native + family arms on CUDA):
  d_k=128, d_v=64, L%64==0 (default L=256, B=4, H=2)

CPU desk run uses fp32 (the reference forwards reject half) with smaller shapes.

Gate: final_loss < 0.5 * initial_loss (+ finite grads via the NaN probe; family arms
also gate on the dispatch counter).
Desk validation: ``--arm eager --steps 60 --device cpu`` achieves the loss criterion;
family arms desk-run the same way (reference-backward fallback stands in off-box).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lethe.kernels.ops.gdn_layer import gdn2_op

# ---------------------------------------------------------------------------
# Delayed-copy task  (delay=1: target[t] = src[t-1])
# ---------------------------------------------------------------------------


def make_batch(
    batch: int, seqlen: int, vocab: int, delay: int, device: torch.device, seed: int
) -> tuple[Tensor, Tensor]:
    """Return (src, tgt) int64 [B, L]. tgt[t] = src[t-delay] when t >= delay, else 0."""
    gen = torch.Generator(device=device).manual_seed(seed)
    src = torch.randint(1, vocab, (batch, seqlen), device=device, generator=gen)
    tgt = torch.zeros_like(src)
    if delay < seqlen:
        tgt[:, delay:] = src[:, : seqlen - delay]
    return src, tgt


# ---------------------------------------------------------------------------
# Family autograd ops — forward = token-serial family oracle; backward = the
# native family dispatch (native_{gla,la,ssd,kda}_backward) with the family's
# autograd reference as the off-box fallback. Dispatch calls are counted so the
# gate can prove the native path actually ran.
# ---------------------------------------------------------------------------

FAMILY_ARMS = ("gla", "la", "ssd", "kda")
_DISPATCH_COUNTS = {"native": 0, "fallback": 0}


def _make_family_op(fam: str) -> Any:
    import lethe.kernels.cute.gdn2_backward as native
    from lethe.kernels.references import family_oracles as fo

    fwd = getattr(fo, f"reference_{fam}_forward")
    ref_bwd = getattr(fo, f"reference_{fam}_backward")
    nat_bwd = getattr(native, f"native_{fam}_backward")

    class _FamilyFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, *tensors: Tensor) -> Tensor:
            with torch.no_grad():
                o = fwd(*tensors)
            ctx.save_for_backward(*tensors)
            return o

        @staticmethod
        def backward(ctx: Any, do: Tensor) -> tuple[Tensor, ...]:
            saved = ctx.saved_tensors
            # Function.backward runs grad-mode-OFF; the native assembly's supporting
            # stages take autograd VJPs internally (stage B, L2-norm VJP).
            with torch.enable_grad():
                grads = nat_bwd(*saved, do.detach())
                if grads is None:
                    _DISPATCH_COUNTS["fallback"] += 1
                    grads = ref_bwd(*saved, do.detach())
                else:
                    _DISPATCH_COUNTS["native"] += 1
            # Grad NamedTuple fields align 1:1 with the family's tensor inputs.
            return tuple(grads)

    def op(*tensors: Tensor) -> Tensor:
        return _FamilyFn.apply(*tensors)  # type: ignore[no-any-return]

    return op


class FamilyMixer(nn.Module):
    """Family-gated mixer: emits exactly the gate set the family mode consumes.

    gla: (q, k, v, g[B,L,H,d_k]) · la: (q, k, v) · ssd: (q, k, v, g[B,L,H]) ·
    kda: (q, k, v, g[B,L,H,d_k], beta[B,L,H]).
    """

    def __init__(
        self, fam: str, d_model: int, nheads: int, d_k: int, d_v: int, dtype: torch.dtype
    ) -> None:
        super().__init__()
        self.fam = fam
        self.nheads = nheads
        self.d_k = d_k
        self.d_v = d_v
        self.dtype = dtype
        self.op = _make_family_op(fam)
        self.q_proj = nn.Linear(d_model, nheads * d_k, bias=False)
        self.k_proj = nn.Linear(d_model, nheads * d_k, bias=False)
        self.v_proj = nn.Linear(d_model, nheads * d_v, bias=False)
        if fam in ("gla", "kda"):
            self.g_proj = nn.Linear(d_model, nheads * d_k, bias=True)  # per key channel
        elif fam == "ssd":
            self.g_proj = nn.Linear(d_model, nheads, bias=True)  # scalar per token-head
        if fam == "kda":
            self.beta_proj = nn.Linear(d_model, nheads, bias=True)
        self.out_proj = nn.Linear(nheads * d_v, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        h, dk, dv = self.nheads, self.d_k, self.d_v
        q = self.q_proj(x).view(b, t, h, dk).to(self.dtype)
        k = self.k_proj(x).view(b, t, h, dk).to(self.dtype)
        v = self.v_proj(x).view(b, t, h, dv).to(self.dtype)
        args: tuple[Tensor, ...]
        if self.fam == "la":
            args = (q, k, v)
        elif self.fam == "ssd":
            g = -F.softplus(self.g_proj(x)).view(b, t, h).to(self.dtype)
            args = (q, k, v, g)
        elif self.fam == "gla":
            g = -F.softplus(self.g_proj(x)).view(b, t, h, dk).to(self.dtype)
            args = (q, k, v, g)
        else:  # kda
            g = -F.softplus(self.g_proj(x)).view(b, t, h, dk).to(self.dtype)
            beta = torch.sigmoid(self.beta_proj(x)).view(b, t, h).to(self.dtype)
            args = (q, k, v, g, beta)
        o = self.op(*args).to(x.dtype)
        return self.out_proj(o.reshape(b, t, h * dv))


# ---------------------------------------------------------------------------
# GDN-2 mixer — genuinely channel-wise gates (crown path, not scalar-reducible)
# ---------------------------------------------------------------------------


class GDN2Mixer(nn.Module):
    def __init__(self, d_model: int, nheads: int, d_k: int, d_v: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.nheads = nheads
        self.d_k = d_k
        self.d_v = d_v
        self.dtype = dtype
        self.q_proj = nn.Linear(d_model, nheads * d_k, bias=False)
        self.k_proj = nn.Linear(d_model, nheads * d_k, bias=False)
        self.v_proj = nn.Linear(d_model, nheads * d_v, bias=False)
        # Per-channel log-decay — softplus output is positive so negating gives log-decay < 0.
        # Each of the nheads*d_k outputs is independent: genuinely channel-wise.
        self.g_proj = nn.Linear(d_model, nheads * d_k, bias=True)
        # Erase/write gates in (0,1) via sigmoid — independent per channel, not scalar.
        self.b_proj = nn.Linear(d_model, nheads * d_k, bias=True)
        self.w_proj = nn.Linear(d_model, nheads * d_v, bias=True)
        self.out_proj = nn.Linear(nheads * d_v, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        h, dk, dv = self.nheads, self.d_k, self.d_v
        q = self.q_proj(x).view(b, t, h, dk)
        k = self.k_proj(x).view(b, t, h, dk)
        v = self.v_proj(x).view(b, t, h, dv)
        g = -F.softplus(self.g_proj(x)).view(b, t, h, dk)
        b_gate = torch.sigmoid(self.b_proj(x)).view(b, t, h, dk)
        w_gate = torch.sigmoid(self.w_proj(x)).view(b, t, h, dv)
        o = gdn2_op(
            q.to(self.dtype),
            k.to(self.dtype),
            v.to(self.dtype),
            g.to(self.dtype),
            b_gate.to(self.dtype),
            w_gate.to(self.dtype),
        ).to(x.dtype)
        return self.out_proj(o.reshape(b, t, h * dv))


class GDN2Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        nheads: int,
        d_k: int,
        d_v: int,
        dtype: torch.dtype,
        fam: str | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mixer: nn.Module = (
            FamilyMixer(fam, d_model, nheads, d_k, d_v, dtype)
            if fam
            else GDN2Mixer(d_model, nheads, d_k, d_v, dtype)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ToyModel(nn.Module):
    def __init__(
        self,
        vocab: int,
        d_model: int,
        nheads: int,
        d_k: int,
        d_v: int,
        n_layers: int,
        dtype: torch.dtype,
        fam: str | None = None,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList(
            [GDN2Block(d_model, nheads, d_k, d_v, dtype, fam=fam) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, src: Tensor) -> Tensor:
        x = self.embed(src)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    arm: str,
    device: torch.device,
    batch: int,
    seqlen: int,
    nheads: int,
    d_k: int,
    d_v: int,
    steps: int,
    seed: int,
    lr: float,
    delay: int,
) -> dict[str, Any]:
    vocab = 32
    n_layers = 2
    # d_model must be divisible by nheads and equal nheads*d_v for square out_proj
    d_model = nheads * d_v

    # fp32 for every arm: bf16 MASTER weights + AdamW diverge (round-B NaN by step
    # 10); fp32 is inside the native dispatch envelope (SUPPORTED_DTYPES) and the
    # tcgen05 kernels still run their f16-operand GEMMs internally.
    run_dtype = torch.float32

    # -----------------------------------------------------------------------
    # Arm: fla
    # -----------------------------------------------------------------------
    if arm == "fla":
        # The fla mixer swap (chunk_gdn2 through the Rosetta A_no_l2norm mapping)
        # is not wired; running the gdn2_op model under this label would fabricate
        # a comparison arm (burst-2 artifact: bit-identical loss to eager).
        print("[fla arm] mixer swap not implemented — skipping (honest no-arm)")
        return {
            "arm": "fla",
            "skipped": True,
            "reason": "fla mixer swap not implemented (Rosetta-mapped wiring pending)",
            "device": str(device),
        }

    # -----------------------------------------------------------------------
    # Arm: eager (monkeypatch is_available -> False)
    # -----------------------------------------------------------------------
    _orig_is_available = None
    if arm == "eager":
        import lethe.kernels.cute.gdn2_backward as _gdn2_shim

        _orig_is_available = _gdn2_shim.is_available
        _gdn2_shim.is_available = lambda *_a, **_kw: False  # type: ignore[assignment]

    # -----------------------------------------------------------------------
    # Arm: assembly (CPU-runnable native glue; kernel refs stand in for tcgen05)
    # -----------------------------------------------------------------------
    _orig_backward = None
    if arm == "assembly":
        import lethe.kernels.ops.gdn_layer as _layer
        from lethe.kernels.cute.gdn2_assemble import (
            assembled_channelwise_gdn2_backward,
        )

        _orig_backward = _layer.gdn2_backward
        _layer.gdn2_backward = assembled_channelwise_gdn2_backward  # type: ignore[assignment]

    fam = arm if arm in FAMILY_ARMS else None
    _DISPATCH_COUNTS["native"] = 0
    _DISPATCH_COUNTS["fallback"] = 0

    try:
        torch.manual_seed(seed)
        model = ToyModel(vocab, d_model, nheads, d_k, d_v, n_layers, run_dtype, fam=fam).to(
            device
        )
        opt = torch.optim.AdamW(model.parameters(), lr=lr)

        loss_curve: list[float] = []
        t0 = time.perf_counter()

        for step in range(steps):
            src, tgt = make_batch(batch, seqlen, vocab, delay, device, seed=step)
            logits = model(src)
            loss = F.cross_entropy(logits.reshape(-1, vocab), tgt.reshape(-1))
            opt.zero_grad()
            loss.backward()
            # NaN probe (round-C: native arm poisons params within 10 steps while
            # eager learns — the arms differ only in the backward). Name the first
            # non-finite grad and its magnitude, then stop with the evidence.
            if step < 15 or step % 50 == 0:
                gmax, bad = 0.0, None
                for pname, prm in model.named_parameters():
                    if prm.grad is None:
                        continue
                    m = prm.grad.abs().max().item()
                    if not math.isfinite(m):
                        bad = pname
                        break
                    gmax = max(gmax, m)
                print(f"[{arm}] step {step:4d} probe loss={float(loss.detach()):.4f} "
                      f"gmax={gmax:.3e} nonfinite={bad}")
                if bad is not None:
                    raise RuntimeError(f"non-finite grad at step {step}: {bad}")
            opt.step()

            if step % 10 == 0:
                v = float(loss.detach())
                loss_curve.append(v)
                print(f"[{arm}] step {step:4d}  loss={v:.4f}")

        wall = time.perf_counter() - t0

    finally:
        if _orig_is_available is not None:
            import lethe.kernels.cute.gdn2_backward as _gdn2_shim2

            _gdn2_shim2.is_available = _orig_is_available  # type: ignore[assignment]
        if _orig_backward is not None:
            import lethe.kernels.ops.gdn_layer as _layer2

            _layer2.gdn2_backward = _orig_backward  # type: ignore[assignment]

    initial = loss_curve[0]
    final = loss_curve[-1]
    gate_ok = final < 0.5 * initial
    dispatch = dict(_DISPATCH_COUNTS)
    if fam and device.type == "cuda":
        # The family purchase requires the NATIVE dispatch to have carried every
        # backward — a fallback-trained curve is not native evidence.
        gate_ok = gate_ok and dispatch["native"] > 0 and dispatch["fallback"] == 0

    result: dict[str, Any] = {
        "arm": arm,
        "device": str(device),
        "torch_version": torch.__version__,
        "steps": steps,
        "loss_curve": loss_curve,
        "initial_loss": initial,
        "final_loss": final,
        "gate_ok": gate_ok,
        "wall_s": round(wall, 2),
        "d_k": d_k,
        "d_v": d_v,
        "seqlen": seqlen,
        "batch": batch,
        "nheads": nheads,
        "run_dtype": str(run_dtype),
        "delay": delay,
        "lr": lr,
    }
    if fam:
        result["backward_dispatch"] = dispatch

    status = "GO" if gate_ok else "FAIL"
    print(
        f"\n[{arm}] {status}: initial={initial:.4f}  final={final:.4f}"
        f"  ratio={final / initial:.3f}  wall={wall:.1f}s"
    )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--arm",
        choices=["native", "fla", "eager", "assembly", *FAMILY_ARMS],
        default="native",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--seqlen", type=int, default=None)
    p.add_argument("--nheads", type=int, default=None)
    p.add_argument("--dv", type=int, default=64, choices=[64, 128])
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--delay", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = torch.device(args.device)

    if args.arm in ("native", *FAMILY_ARMS) and device.type == "cuda":
        # Enforce the dispatch envelope
        d_k = 128
        d_v = args.dv
        batch = args.batch if args.batch is not None else 4
        seqlen = args.seqlen if args.seqlen is not None else 256
        nheads = args.nheads if args.nheads is not None else 2
        lr = args.lr if args.lr is not None else 1e-2
        if seqlen % 64 != 0:
            raise ValueError(f"seqlen {seqlen} must be divisible by 64 for native arm")
    else:
        # CPU / eager: smaller shapes for speed; d_k/d_v kept small
        d_k = 32
        d_v = 32
        batch = args.batch if args.batch is not None else 4
        seqlen = args.seqlen if args.seqlen is not None else 32
        nheads = args.nheads if args.nheads is not None else 2
        lr = args.lr if args.lr is not None else 1e-2

    result = train(
        arm=args.arm,
        device=device,
        batch=batch,
        seqlen=seqlen,
        nheads=nheads,
        d_k=d_k,
        d_v=d_v,
        steps=args.steps,
        seed=args.seed,
        lr=lr,
        delay=args.delay,
    )

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results -> {args.out}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
