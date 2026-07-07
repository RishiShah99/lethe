"""Phase-1 KILL-GATE: cross-check the GDN-2 oracle against fla on Hopper/H100.

Runs on the GPU box (fla compiles cleanly on Hopper). Validates
``reference_gdn2_forward`` two ways, on >= 8 shapes, and checks determinism:

1. SCALAR REDUCTION vs fla ``naive_recurrent_gated_delta_rule`` (pure torch):
   set b = w = beta and a per-head (channel-constant) g, with q/k L2-normed (the
   regime GDN-2 actually runs in). Normalization is REQUIRED for a meaningful
   comparison: the delta rule's contraction factor is ``1 - beta*||k||^2``, so raw
   randn k (||k||^2 ~ d_k) makes the recurrence divergent and any two
   implementations separate by chaos (rounding amplified to overflow), not by a
   correctness gap. fla naive does not L2-norm internally, so we pre-normalize and
   feed identical q/k to both sides. Our oracle must match fla to MACHINE PRECISION
   in fp64 (the rigorous gate) and to ~1e-4 in fp32.
2. SCALAR vs fla ``chunk_gated_delta_rule`` (the real Triton kernel) where the
   API is available: kernel tolerance, the on-hardware cross-check.
3. DETERMINISM: the oracle run twice is bitwise-equal.

Also records the GDN-2 channel-wise self-consistency on GPU (b=w=beta path equals
the scalar path) — the built-in correctness test, already CPU-proven.

GO iff the fp64 scalar reduction matches fla on every shape AND determinism holds.
Writes results/gdn2_hopper_xcheck.json. Run:
    CUDA_HOME=/usr/local/cuda-13.0 PATH=$CUDA_HOME/bin:$PATH \
    uv run --no-sync python -m scratch.gdn2_hopper_xcheck
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

import torch

from lethe.kernels.references.gdn_backward import reference_gdn2_forward


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)

# (batch, seqlen, nheads, d_k, d_v) — >= 8 shapes spanning the grid.
SHAPES: list[tuple[int, int, int, int, int]] = [
    (1, 64, 2, 16, 16),
    (2, 128, 4, 32, 32),
    (1, 256, 8, 64, 64),
    (2, 512, 4, 128, 128),
    (1, 512, 16, 128, 128),
    (4, 256, 8, 64, 128),
    (2, 1024, 2, 128, 128),
    (1, 2048, 4, 64, 64),
    (3, 128, 6, 96, 96),
]


def _make(shape: tuple[int, int, int, int, int], dtype: torch.dtype, dev: str, seed: int):
    b, t, h, dk, dv = shape
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(b, t, h, dk, generator=gen, dtype=torch.float64)
    k = torch.randn(b, t, h, dk, generator=gen, dtype=torch.float64)
    v = torch.randn(b, t, h, dv, generator=gen, dtype=torch.float64)
    dt = torch.exp(torch.rand(b, t, h, generator=gen, dtype=torch.float64) * 4.6 - 6.9)  # ~[1e-3,1e-1]
    a_head = -torch.rand(h, generator=gen, dtype=torch.float64)
    g_head = dt * a_head  # [b,t,h] <= 0
    beta = torch.rand(b, t, h, generator=gen, dtype=torch.float64) * 0.8 + 0.1
    cast = lambda x: x.to(device=dev, dtype=dtype)  # noqa: E731
    return tuple(cast(x) for x in (q, k, v, g_head, beta))


def _independent_gdn_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Independent reimpl of fla's gated-delta naive math (distinct reduction order
    from the oracle), kept in the input dtype — the machine-precision external
    anchor. fla's own naive casts every input to fp32 (``naive.py`` L40), so it can
    only confirm the algorithm to fp32 precision, never serve as the fp64 gold tier.
    """
    bsz, seqlen, nh, dk = k.shape
    dv = v.shape[-1]
    qh, kh, vh = (x.transpose(1, 2) for x in (q, k, v))
    bh, gh = beta.transpose(1, 2), g.transpose(1, 2)
    qh = qh * scale
    s = torch.zeros(bsz, nh, dk, dv, dtype=q.dtype, device=q.device)
    out = []
    for tt in range(seqlen):
        s = s * gh[:, :, tt].exp()[..., None, None]
        kt = kh[:, :, tt]
        kv = (s * kt[..., :, None]).sum(-2)
        delta = (vh[:, :, tt] - kv) * bh[:, :, tt][..., None]
        s = s + kt[..., :, None] * delta[..., None, :]
        out.append((s * qh[:, :, tt][..., :, None]).sum(-2))
    return torch.stack(out, dim=2).transpose(1, 2).contiguous()


def _scalar_reduction_check(dev: str) -> dict[str, Any]:
    """Oracle (b=w=beta, per-head g, q/k L2-normed) vs TWO references: an independent
    fp64 reimpl (machine-precision gate) and fla's naive kernel (fp32 floor, since fla
    casts to fp32). Both must pass for GO."""
    from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule

    rows: list[dict[str, Any]] = []
    # (dtype, independent machine-precision tol, fla fp32-floor tol)
    for dtype, indep_tol, fla_tol in (
        (torch.float64, 1e-9, 1e-5),
        (torch.float32, 1e-4, 1e-4),
    ):
        for i, shape in enumerate(SHAPES):
            b, t, h, dk, dv = shape
            if dv != dk:
                continue  # fla scalar path assumes a single head dim per head
            q, k, v, g_head, beta = _make(shape, dtype, dev, seed=100 + i)
            scale = dk**-0.5
            qn, kn = _l2norm(q), _l2norm(k)  # stable regime; identical inputs to all
            g_chan = g_head.unsqueeze(-1).expand(b, t, h, dk).contiguous()
            b_gate = beta.unsqueeze(-1).expand(b, t, h, dk).contiguous()
            w_gate = beta.unsqueeze(-1).expand(b, t, h, dv).contiguous()
            ours = reference_gdn2_forward(
                qn, kn, v, g_chan, b_gate, w_gate, scale=scale, use_qk_l2norm=False
            )
            indep = _independent_gdn_naive(qn, kn, v, beta, g_head, scale)
            o_fla, _ = naive_recurrent_gated_delta_rule(
                qn, kn, v, beta, g_head, scale=scale, output_final_state=False
            )
            err_indep = (ours - indep).abs().max().item()
            err_fla = (ours - o_fla.to(ours.dtype)).abs().max().item()
            rows.append(
                {
                    "shape": shape,
                    "dtype": str(dtype),
                    "err_indep": err_indep,
                    "indep_tol": indep_tol,
                    "err_fla": err_fla,
                    "fla_tol": fla_tol,
                    "passed": err_indep <= indep_tol and err_fla <= fla_tol,
                }
            )
    return {"available": True, "rows": rows, "all_passed": all(r["passed"] for r in rows)}


def _chunk_kernel_check(dev: str) -> dict[str, Any]:
    """Our oracle vs fla chunk_gated_delta_rule (Triton kernel), bf16/fp32."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    rows: list[dict[str, Any]] = []
    for dtype, tol in ((torch.float32, 2e-2), (torch.bfloat16, 8e-2)):
        for i, shape in enumerate(SHAPES):
            b, t, h, dk, dv = shape
            if dv != dk:
                continue
            q, k, v, g_head, beta = _make(shape, dtype, dev, seed=200 + i)
            scale = dk**-0.5
            g_chan = g_head.unsqueeze(-1).expand(b, t, h, dk).contiguous()
            b_gate = beta.unsqueeze(-1).expand(b, t, h, dk).contiguous()
            w_gate = beta.unsqueeze(-1).expand(b, t, h, dv).contiguous()
            ours = reference_gdn2_forward(
                q.double(), k.double(), v.double(), g_chan.double(),
                b_gate.double(), w_gate.double(), scale=scale, use_qk_l2norm=True,
            )
            try:
                o_fla, _ = chunk_gated_delta_rule(
                    q, k, v, g_head, beta, scale=scale,
                    use_qk_l2norm_in_kernel=True, output_final_state=False,
                )
            except TypeError:
                o_fla, _ = chunk_gated_delta_rule(q, k, v, g_head, beta, scale=scale)
            scale_inf = max(1.0, o_fla.double().abs().max().item())
            max_err = (ours - o_fla.double()).abs().max().item() / scale_inf
            rows.append(
                {"shape": shape, "dtype": str(dtype), "max_err_rel": max_err,
                 "tol": tol, "passed": max_err <= tol}
            )
    return {"available": True, "rows": rows, "all_passed": all(r["passed"] for r in rows)}


def _determinism_check(dev: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for i, shape in enumerate(SHAPES):
        b, t, h, dk, dv = shape
        q, k, v, g_head, beta = _make(shape, torch.float32, dev, seed=300 + i)
        g_chan = g_head.unsqueeze(-1).expand(b, t, h, dk).contiguous()
        b_gate = beta.unsqueeze(-1).expand(b, t, h, dk).contiguous()
        w_gate = beta.unsqueeze(-1).expand(b, t, h, dv).contiguous()
        o1 = reference_gdn2_forward(q, k, v, g_chan, b_gate, w_gate)
        o2 = reference_gdn2_forward(q, k, v, g_chan, b_gate, w_gate)
        rows.append({"shape": shape, "bitwise_equal": bool(torch.equal(o1, o2))})
    return {"all_passed": all(r["bitwise_equal"] for r in rows), "rows": rows}


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out: dict[str, Any] = {
        "device": dev,
        "cuda_name": torch.cuda.get_device_name(0) if dev == "cuda" else None,
    }
    for name, fn in (
        ("scalar_reduction_vs_fla_naive", _scalar_reduction_check),
        ("chunk_kernel_vs_fla", _chunk_kernel_check),
        ("determinism", _determinism_check),
    ):
        try:
            out[name] = fn(dev)
        except ImportError as exc:
            out[name] = {"available": False, "skipped": f"ImportError: {exc}"}
        except Exception as exc:
            out[name] = {"available": False, "error": f"{type(exc).__name__}: {exc}",
                         "trace": traceback.format_exc()}

    scalar = out.get("scalar_reduction_vs_fla_naive", {})
    determ = out.get("determinism", {})
    out["GO"] = bool(scalar.get("all_passed")) and bool(determ.get("all_passed"))

    dest = Path("results/gdn2_hopper_xcheck.json")
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: (v.get("all_passed") if isinstance(v, dict) else v)
                      for k, v in out.items()}, indent=2))
    print(f"\nGO={out['GO']}  ->  {dest}")


if __name__ == "__main__":
    main()
