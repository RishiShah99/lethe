"""Positive control for the audit_harness rigor-gap rebuttal (HANDOFF Task 2).

Runs our own known-good kernels through the SAME ``audit_harness.audit_worker``
battery the Dr.Kernel corpus audit uses, wrapped in the identical KernelBench
``Model``/``get_inputs``/``get_init_inputs``/``ModelNew`` source-string convention
(``scratch/audit_extract_drkernel.py``'s row format). Each row pairs a torch
reference (``Model``) against our Triton/native op (``ModelNew``); auxiliary
operands are fixed weights baked in via ``get_init_inputs`` (registered as
buffers so the harness's mixed-precision round-trip applies to them), and
``get_inputs`` returns only the single tensor the gates drive (matching how the
corpus's own KernelBench rows are structured).

Shapes are cited against ``verifier/op_harness.py``'s own calibrated constants
(never guessed) and against ``kernels/cute/gdn2_backward.py``'s crown tile dims
for the GDN-2 row. Expected: near-100% pass; any gate failure here is a real
bug in one of OUR kernels or references, not evidence the referee flatters
itself. Runs one sandboxed subprocess per kernel via ``run_in_subprocess``,
exactly as ``scratch/audit_run.py`` does for the corpus rows.

Usage:
    uv run python scratch/positive_control.py --device cuda \
        --json results/positive_control.json
    uv run python scratch/positive_control.py --device cpu --only C1,C2
    uv run python scratch/positive_control.py --self-test
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from typing import Any

from lethe.verifier.audit_harness import AUDIT_GATE_NAMES, GATE_SHORT_NAMES
from lethe.verifier.sandbox import run_in_subprocess

_MODULE = "lethe.verifier.audit_harness"
_FUNC = "audit_worker"
_GATE_ORDER: tuple[str, ...] = tuple(GATE_SHORT_NAMES[name] for name in AUDIT_GATE_NAMES)

# Shape constants, cited against the calibrated defaults elsewhere in the
# verifier rather than guessed:
_SCAN_N_STATE = 16  # op_harness.py:81  SCAN_N_STATE
_SCAN_CHUNK = 8  # op_harness.py:84  SCAN_CHUNK_SIZE
_MIMO_HEADDIM = 4  # op_harness.py:573 MIMO_HEADDIM
_MIMO_RANK = 4  # op_harness.py:574 MIMO_RANK
_MIMO_N_STATE = 16  # op_harness.py:575 MIMO_N_STATE
_ROPE_N_STATE = 16  # op_harness.py:1526 ROPE_N_STATE
_ROPE_NUM_ANGLES = 6  # op_harness.py:1529 ROPE_NUM_ANGLES
_FUSED_CONV_K = 4  # op_harness.py:1682 FUSED_CONV_K
_GDN_D_K = 128  # cute/gdn2_backward.py:59 _KERNEL_D_K
_GDN_D_V = 64  # cute/gdn2_backward.py:60 _KERNEL_D_V (also in _KERNEL_D_V_CW)
_GDN_CHUNK = 64  # cute/gdn2_backward.py:61 _KERNEL_CHUNK


def _c1_sources() -> tuple[str, str]:
    """C1 forward chunked scan: u primary [B,L,D]; delta/A/B/C/D fixed aux."""
    batch, seq, d_model, n_state, chunk = 2, 64, 32, _SCAN_N_STATE, _SCAN_CHUNK
    body = f"""\
def get_init_inputs():
    delta = torch.randn({batch}, {seq}, {d_model})
    A = -torch.rand({d_model}, {n_state})
    B = torch.randn({batch}, {seq}, {n_state})
    C = torch.randn({batch}, {seq}, {n_state})
    D = torch.randn({d_model})
    return [delta, A, B, C, D]


def get_inputs():
    return [torch.randn({batch}, {seq}, {d_model})]


class {{cls}}(nn.Module):
    def __init__(self, delta, A, B, C, D):
        super().__init__()
        self.register_buffer("delta", delta)
        self.register_buffer("A", A)
        self.register_buffer("B", B)
        self.register_buffer("C", C)
        self.register_buffer("D", D)

    def forward(self, u):
        return {{call}}(u, self.delta, self.A, self.B, self.C, self.D, chunk_size={chunk})
"""
    ref = (
        "import torch\nfrom torch import nn\n\n"
        "from lethe.kernels.references import reference_forward_chunked_scan\n\n"
        + body.format(cls="Model", call="reference_forward_chunked_scan")
    )
    cand = (
        "import torch\nfrom torch import nn\n\n"
        "from lethe.kernels.ops import forward_chunked_scan\n\n"
        + body.format(cls="ModelNew", call="forward_chunked_scan")
    )
    return ref, cand


def _c2_sources() -> tuple[str, str]:
    """C2 backward selective scan: dy primary [B,L,D]; u/delta/A/B/C/D fixed aux."""
    batch, seq, d_model, n_state, chunk = 2, 64, 32, _SCAN_N_STATE, _SCAN_CHUNK
    body = f"""\
def get_init_inputs():
    u = torch.randn({batch}, {seq}, {d_model})
    delta = torch.randn({batch}, {seq}, {d_model})
    A = -torch.rand({d_model}, {n_state})
    B = torch.randn({batch}, {seq}, {n_state})
    C = torch.randn({batch}, {seq}, {n_state})
    D = torch.randn({d_model})
    return [u, delta, A, B, C, D]


def get_inputs():
    return [torch.randn({batch}, {seq}, {d_model})]


class {{cls}}(nn.Module):
    def __init__(self, u, delta, A, B, C, D):
        super().__init__()
        self.register_buffer("u", u)
        self.register_buffer("delta", delta)
        self.register_buffer("A", A)
        self.register_buffer("B", B)
        self.register_buffer("C", C)
        self.register_buffer("D", D)

    def forward(self, dy):
        return {{call}}(
            self.u, self.delta, self.A, self.B, self.C, self.D, dy, chunk_size={chunk}
        )
"""
    ref = (
        "import torch\nfrom torch import nn\n\n"
        "from lethe.kernels.references import reference_backward_selective_scan\n\n"
        + body.format(cls="Model", call="reference_backward_selective_scan")
    )
    cand = (
        "import torch\nfrom torch import nn\n\n"
        "from lethe.kernels.ops import backward_selective_scan\n\n"
        + body.format(cls="ModelNew", call="backward_selective_scan")
    )
    return ref, cand


def _c3_sources() -> tuple[str, str]:
    """C3 MIMO backward: dy primary [B,L,H,P]; x/B/C/dt/alpha/mimo_x/mimo_o fixed aux."""
    batch, seq = 2, 16
    nheads, headdim, rank, n_state = _MIMO_HEADDIM, 4, _MIMO_RANK, _MIMO_N_STATE
    body = f"""\
def get_init_inputs():
    x = torch.randn({batch}, {seq}, {nheads}, {headdim})
    B = torch.randn({batch}, {seq}, {rank}, {nheads}, {n_state})
    C = torch.randn({batch}, {seq}, {rank}, {nheads}, {n_state})
    dt = 0.01 + 0.09 * torch.rand({batch}, {seq}, {nheads})
    a_head = -torch.rand({nheads})
    alpha = torch.exp(dt * a_head)
    mimo_x = 1.0 / {rank} + 0.1 * torch.randn({nheads}, {rank}, {headdim})
    mimo_o = 1.0 / {rank} + 0.1 * torch.randn({nheads}, {rank}, {headdim})
    return [x, B, C, dt, alpha, mimo_x, mimo_o]


def get_inputs():
    return [torch.randn({batch}, {seq}, {nheads}, {headdim})]


class {{cls}}(nn.Module):
    def __init__(self, x, B, C, dt, alpha, mimo_x, mimo_o):
        super().__init__()
        self.register_buffer("x", x)
        self.register_buffer("B", B)
        self.register_buffer("C", C)
        self.register_buffer("dt", dt)
        self.register_buffer("alpha", alpha)
        self.register_buffer("mimo_x", mimo_x)
        self.register_buffer("mimo_o", mimo_o)

    def forward(self, dy):
        return {{call}}(
            self.x, self.B, self.C, self.dt, self.alpha, self.mimo_x, self.mimo_o, dy
        )
"""
    ref = (
        "import torch\nfrom torch import nn\n\n"
        "from lethe.kernels.references import reference_mimo_backward\n\n"
        + body.format(cls="Model", call="reference_mimo_backward")
    )
    cand = (
        "import torch\nfrom torch import nn\n\n"
        "from lethe.kernels.ops import mimo_backward\n\n"
        + body.format(cls="ModelNew", call="mimo_backward")
    )
    return ref, cand


def _c4_sources() -> tuple[str, str]:
    """C4 complex-RoPE scan: x primary [B,L,H,P]; B/C/dt/A/angle_proj fixed aux."""
    batch, seq, nheads, headdim = 2, 16, 4, 4
    n_state, num_angles = _ROPE_N_STATE, _ROPE_NUM_ANGLES
    body = f"""\
def get_init_inputs():
    B = torch.randn({batch}, {seq}, {nheads}, {n_state})
    C = torch.randn({batch}, {seq}, {nheads}, {n_state})
    dt = 0.01 + 0.09 * torch.rand({batch}, {seq}, {nheads})
    A = -torch.rand({nheads})
    angle_proj = torch.randn({batch}, {seq}, {nheads}, {num_angles})
    return [B, C, dt, A, angle_proj]


def get_inputs():
    return [torch.randn({batch}, {seq}, {nheads}, {headdim})]


class {{cls}}(nn.Module):
    def __init__(self, B, C, dt, A, angle_proj):
        super().__init__()
        self.register_buffer("B", B)
        self.register_buffer("C", C)
        self.register_buffer("dt", dt)
        self.register_buffer("A", A)
        self.register_buffer("angle_proj", angle_proj)

    def forward(self, x):
        return {{call}}(x, self.B, self.C, self.dt, self.A, self.angle_proj)
"""
    ref = (
        "import torch\nfrom torch import nn\n\n"
        "from lethe.kernels.references import reference_complex_scan_rope\n\n"
        + body.format(cls="Model", call="reference_complex_scan_rope")
    )
    cand = (
        "import torch\nfrom torch import nn\n\n"
        "from lethe.kernels.ops import complex_scan_rope\n\n"
        + body.format(cls="ModelNew", call="complex_scan_rope")
    )
    return ref, cand


def _c5_sources() -> tuple[str, str]:
    """C5 fused block forward: padded x primary [B,L_pad,D]; scan/conv/norm fixed aux."""
    batch, l_out, d_model, n_state = 2, 64, 32, _SCAN_N_STATE
    conv_k, chunk = _FUSED_CONV_K, _SCAN_CHUNK
    l_pad = l_out + conv_k - 1
    body = f"""\
def get_init_inputs():
    conv_weight = torch.randn({d_model}, 1, {conv_k}) / {conv_k} ** 0.5
    conv_bias = 0.5 * torch.randn({d_model})
    delta = torch.randn({batch}, {l_out}, {d_model})
    A = -torch.rand({d_model}, {n_state})
    B = torch.randn({batch}, {l_out}, {n_state})
    C = torch.randn({batch}, {l_out}, {n_state})
    D = torch.randn({d_model})
    norm_weight = 1.0 + 0.25 * torch.randn({d_model})
    return [conv_weight, conv_bias, delta, A, B, C, D, norm_weight]


def get_inputs():
    return [torch.randn({batch}, {l_pad}, {d_model})]


class {{cls}}(nn.Module):
    def __init__(self, conv_weight, conv_bias, delta, A, B, C, D, norm_weight):
        super().__init__()
        self.register_buffer("conv_weight", conv_weight)
        self.register_buffer("conv_bias", conv_bias)
        self.register_buffer("delta", delta)
        self.register_buffer("A", A)
        self.register_buffer("B", B)
        self.register_buffer("C", C)
        self.register_buffer("D", D)
        self.register_buffer("norm_weight", norm_weight)

    def forward(self, x):
        return {{call}}(
            x,
            self.conv_weight,
            self.conv_bias,
            self.delta,
            self.A,
            self.B,
            self.C,
            self.D,
            self.norm_weight,
            conv_kernel_size={conv_k},
            chunk_size={chunk},
        )
"""
    ref = (
        "import torch\nfrom torch import nn\n\n"
        "from lethe.kernels.references import reference_fused_block_forward\n\n"
        + body.format(cls="Model", call="reference_fused_block_forward")
    )
    cand = (
        "import torch\nfrom torch import nn\n\n"
        "from lethe.kernels.ops import fused_block_forward\n\n"
        + body.format(cls="ModelNew", call="fused_block_forward")
    )
    return ref, cand


def _c6_sources() -> tuple[str, str]:
    """C6 fused block backward: dy primary [B,L_out,D]; x(padded)+scan/conv/norm fixed aux."""
    batch, l_out, d_model, n_state = 2, 64, 32, _SCAN_N_STATE
    conv_k, chunk = _FUSED_CONV_K, _SCAN_CHUNK
    l_pad = l_out + conv_k - 1
    body = f"""\
def get_init_inputs():
    x = torch.randn({batch}, {l_pad}, {d_model})
    conv_weight = torch.randn({d_model}, 1, {conv_k}) / {conv_k} ** 0.5
    conv_bias = 0.5 * torch.randn({d_model})
    delta = torch.randn({batch}, {l_out}, {d_model})
    A = -torch.rand({d_model}, {n_state})
    B = torch.randn({batch}, {l_out}, {n_state})
    C = torch.randn({batch}, {l_out}, {n_state})
    D = torch.randn({d_model})
    norm_weight = 1.0 + 0.25 * torch.randn({d_model})
    return [x, conv_weight, conv_bias, delta, A, B, C, D, norm_weight]


def get_inputs():
    return [torch.randn({batch}, {l_out}, {d_model})]


class {{cls}}(nn.Module):
    def __init__(self, x, conv_weight, conv_bias, delta, A, B, C, D, norm_weight):
        super().__init__()
        self.register_buffer("x", x)
        self.register_buffer("conv_weight", conv_weight)
        self.register_buffer("conv_bias", conv_bias)
        self.register_buffer("delta", delta)
        self.register_buffer("A", A)
        self.register_buffer("B", B)
        self.register_buffer("C", C)
        self.register_buffer("D", D)
        self.register_buffer("norm_weight", norm_weight)

    def forward(self, dy):
        return {{call}}(
            self.x,
            self.conv_weight,
            self.conv_bias,
            self.delta,
            self.A,
            self.B,
            self.C,
            self.D,
            self.norm_weight,
            dy,
            conv_kernel_size={conv_k},
            chunk_size={chunk},
        )
"""
    ref = (
        "import torch\nfrom torch import nn\n\n"
        "from lethe.kernels.references import reference_fused_block_backward\n\n"
        + body.format(cls="Model", call="reference_fused_block_backward")
    )
    cand = (
        "import torch\nfrom torch import nn\n\n"
        "from lethe.kernels.ops import fused_block_backward\n\n"
        + body.format(cls="ModelNew", call="fused_block_backward")
    )
    return ref, cand


def _gdn_sources() -> tuple[str, str]:
    """GDN-2 native backward: do primary [B,L,H,d_v]; q/k/v/g/b/w fixed aux.

    Crown tile dims (d_k=128, d_v=64, L a multiple of 64) so the candidate's
    internal dispatch (``lethe.kernels.ops.gdn2_backward``) routes to the
    channel-wise tcgen05 kernel on a Blackwell box instead of its eager
    fallback; g/b/w vary per channel (not scalar-reducible) to avoid the
    scalar (Phase-2) route.
    """
    batch, seq, nheads = 1, _GDN_CHUNK, 1
    d_k, d_v = _GDN_D_K, _GDN_D_V
    body = f"""\
def get_init_inputs():
    q = torch.randn({batch}, {seq}, {nheads}, {d_k})
    k = torch.randn({batch}, {seq}, {nheads}, {d_k})
    v = torch.randn({batch}, {seq}, {nheads}, {d_v})
    g = -0.1 * torch.rand({batch}, {seq}, {nheads}, {d_k})
    b = torch.randn({batch}, {seq}, {nheads}, {d_k}).sigmoid()
    w = torch.randn({batch}, {seq}, {nheads}, {d_v}).sigmoid()
    return [q, k, v, g, b, w]


def get_inputs():
    return [torch.randn({batch}, {seq}, {nheads}, {d_v})]


class {{cls}}(nn.Module):
    def __init__(self, q, k, v, g, b, w):
        super().__init__()
        self.register_buffer("q", q)
        self.register_buffer("k", k)
        self.register_buffer("v", v)
        self.register_buffer("g", g)
        self.register_buffer("b", b)
        self.register_buffer("w", w)

    def forward(self, do):
        return {{call}}(self.q, self.k, self.v, self.g, self.b, self.w, do)
"""
    ref = (
        "import torch\nfrom torch import nn\n\n"
        "from lethe.kernels.references import reference_gdn2_backward\n\n"
        + body.format(cls="Model", call="reference_gdn2_backward")
    )
    cand = (
        "import torch\nfrom torch import nn\n\n"
        "from lethe.kernels.ops import gdn2_backward\n\n"
        + body.format(cls="ModelNew", call="gdn2_backward")
    )
    return ref, cand


_KERNEL_BUILDERS: dict[str, Callable[[], tuple[str, str]]] = {
    "C1": _c1_sources,
    "C2": _c2_sources,
    "C3": _c3_sources,
    "C4": _c4_sources,
    "C5": _c5_sources,
    "C6": _c6_sources,
    "GDN": _gdn_sources,
}

_SELFTEST_REF = """\
import torch
from torch import nn


def get_inputs():
    return [torch.randn(4, 8)]


class Model(nn.Module):
    def forward(self, x):
        return x * 2.0
"""

_SELFTEST_CAND = """\
import torch
from torch import nn


def get_inputs():
    return [torch.randn(4, 8)]


class ModelNew(nn.Module):
    def forward(self, x):
        return x * 2.0
"""


def _run_audit(ref_source: str, cand_source: str, device: str, timeout_s: float) -> dict[str, Any]:
    res = run_in_subprocess(
        _MODULE,
        _FUNC,
        (ref_source, cand_source, {"device": device}),
        timeout_s=timeout_s,
        memory_limit_mb=0,
    )
    if res.success and isinstance(res.output, dict):
        return dict(res.output)
    return {"status": f"sandbox_{res.error_class.name.lower()}", "error": res.stderr[-400:]}


def _kernel_verdict(result: dict[str, Any]) -> tuple[bool, list[str]]:
    """(all_pass, fail_entries) for one kernel's raw audit_worker result."""
    if result.get("status") != "gated":
        return False, [f"{result.get('status')}: {result.get('error', '')}"]
    fails = []
    for gate, info in result.get("gates", {}).items():
        if info["status"] not in ("pass", "na"):
            fails.append(f"{gate}={info['status']} ({info.get('reason', '')})")
    return not fails, fails


def _print_table(results: dict[str, dict[str, Any]]) -> None:
    header = f"{'kernel':<8}" + "".join(f"{g:<9}" for g in _GATE_ORDER)
    print(header)
    print("-" * len(header))
    for name, result in results.items():
        if result.get("status") == "gated":
            gates = result["gates"]
            row = f"{name:<8}" + "".join(f"{gates[g]['status']:<9}" for g in _GATE_ORDER)
        else:
            row = f"{name:<8}{result.get('status', 'unknown')}"
        print(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", default="results/positive_control.json")
    ap.add_argument(
        "--only",
        "--kernels",
        dest="kernels",
        default=None,
        help="comma-separated subset, e.g. C1,C2",
    )
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--self-test", action="store_true", help="tiny CPU identity plumbing check")
    args = ap.parse_args()

    if args.self_test:
        result = _run_audit(_SELFTEST_REF, _SELFTEST_CAND, "cpu", args.timeout)
        print(json.dumps(result, indent=2))
        ok, fails = _kernel_verdict(result)
        print("SELF-TEST", "PASS" if ok else f"FAIL {fails}")
        return

    names = list(_KERNEL_BUILDERS)
    if args.kernels:
        wanted = {k.strip() for k in args.kernels.split(",")}
        names = [n for n in names if n in wanted]
    if args.device != "cuda" and "GDN" in names:
        names = [n for n in names if n != "GDN"]
        print("[positive_control] skipping GDN: B200-only tcgen05 kernel, device != cuda")

    results: dict[str, dict[str, Any]] = {}
    for name in names:
        ref_source, cand_source = _KERNEL_BUILDERS[name]()
        results[name] = _run_audit(ref_source, cand_source, args.device, args.timeout)

    _print_table(results)

    gate_fails: list[str] = []
    all_pass = True
    for name, result in results.items():
        ok, fails = _kernel_verdict(result)
        all_pass = all_pass and ok
        gate_fails.extend(f"{name}:{f}" for f in fails)

    summary = {"n_kernels": len(results), "all_pass": all_pass, "gate_fails": gate_fails}
    out = {"kernels": results, "summary": summary}

    out_dir = os.path.dirname(args.json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
