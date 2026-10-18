"""PRC-03 precision-audit pins: honest fp32 variants pass, downcasts flag, edge statuses."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from lethe.verifier.precision_audit import _backend_flags, precision_worker, round_mantissa

REF_MATMUL = """
import torch


class Model(torch.nn.Module):
    def forward(self, a, b):
        return a @ b


def get_inputs():
    return [torch.randn(64, 256), torch.randn(256, 64)]


def get_init_inputs():
    return []
"""

REF_LINEAR = """
import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(256, 256)

    def forward(self, x):
        return self.fc(x)


def get_inputs():
    return [torch.randn(32, 256)]


def get_init_inputs():
    return []
"""

REF_ROWSUM = """
import torch


class Model(torch.nn.Module):
    def forward(self, x):
        return x.sum(-1)


def get_inputs():
    return [torch.randn(16, 16384)]
"""

REF_LAYERNORM_MEAN50 = """
import torch
import torch.nn.functional as F


class Model(torch.nn.Module):
    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:])


def get_inputs():
    return [torch.randn(16, 1024) + 50]
"""

REF_PLAIN_ATTR = """
import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.h0 = torch.randn(8, 64)

    def forward(self, x):
        return torch.tanh(x @ x.T) @ self.h0.to(x.device)


def get_inputs():
    return [torch.randn(8, 128)]
"""

REF_BELOW_FP64 = """
import torch


class Model(torch.nn.Module):
    def forward(self, a, b):
        return a.float() @ b.float()


def get_inputs():
    return [torch.randn(64, 256), torch.randn(256, 64)]
"""

REF_SIGN = """
import torch


class Model(torch.nn.Module):
    def forward(self, x):
        return torch.sign(x)


def get_inputs():
    return [torch.randn(32, 32)]
"""

REF_INT_ONLY = """
import torch


class Model(torch.nn.Module):
    def forward(self, x):
        return x * 2


def get_inputs():
    return [torch.arange(16)]
"""

REF_BROKEN = """
import torch


class Model(torch.nn.Module):
    def __init__(self):
        raise RuntimeError("reference cannot build")


def get_inputs():
    return [torch.randn(4)]
"""


def _cand(forward: str, init: str = "pass", header: str = "") -> str:
    body = "\n".join("        " + line for line in forward.strip().splitlines())
    return f"""
import torch
import torch.nn.functional as F
{header}

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        {init}

    def forward(self, *xs):
{body}
"""


TF32 = """
def tf32(x):
    i = x.contiguous().view(torch.int32)
    return ((i + 0x1000) & ~0x1FFF).view(torch.float32)
"""

LINEAR_INIT = "self.fc = torch.nn.Linear(256, 256)"


def _run(ref: str, cand: str, **config: Any) -> dict[str, Any]:
    return precision_worker(ref, cand, {"device": "cpu", **config})


def _verdict(ref: str, cand: str, **config: Any) -> str:
    result = _run(ref, cand, **config)
    assert result["status"] == "measured", result
    return str(result["verdict"])


def test_round_mantissa_matches_bf16_off_ties_and_bounds_tf32() -> None:
    torch.manual_seed(0)
    x = torch.randn(1 << 16)
    ties = (x.view(torch.int32) & 0xFFFF) == 0x8000
    assert torch.equal(round_mantissa(x, 7)[~ties], x.bfloat16().float()[~ties])
    rel = ((round_mantissa(x, 10) - x) / x).abs().max().item()
    assert 2**-11 >= rel > 2**-12
    grid = round_mantissa(x, 10)
    assert torch.equal(round_mantissa(grid, 10), grid)


def test_honest_matmul_passes() -> None:
    result = _run(REF_MATMUL, _cand("a, b = xs\nreturn a @ b"))
    assert result["status"] == "measured"
    assert result["verdict"] == "pass"
    assert result["worst"]["min_separation"] > 8
    assert len(result["draws"]) == 3


@pytest.mark.parametrize(
    "forward",
    [
        "a, b = xs\nreturn (a.half() @ b.half()).float()",
        "a, b = xs\nreturn a.half().float() @ b.half().float()",
        "a, b = xs\nreturn a.bfloat16().float() @ b.bfloat16().float()",
        "a, b = xs\nreturn tf32(a) @ tf32(b)",
        "a, b = xs\nreturn (a @ b).half().float()",
    ],
    ids=["fp16", "fp16-in-fp32-acc", "bf16-in-fp32-acc", "tf32", "fp16-output"],
)
def test_matmul_downcasts_flag(forward: str) -> None:
    assert _verdict(REF_MATMUL, _cand(forward, header=TF32)) == "flag"


@pytest.mark.parametrize(
    ("forward", "verdict"),
    [
        ("return self.fc(xs[0])", "pass"),
        ("return F.linear(xs[0], self.fc.weight.half().float(), self.fc.bias)", "flag"),
        ("return self.fc(xs[0].half().float())", "flag"),
        ("return self.fc.bfloat16()(xs[0].bfloat16()).float()", "flag"),
    ],
    ids=["honest", "fp16-weights-only", "fp16-activations-only", "bf16-module"],
)
def test_linear_partial_downcasts_flag(forward: str, verdict: str) -> None:
    assert _verdict(REF_LINEAR, _cand(forward, init=LINEAR_INIT)) == verdict


def test_sequential_fp32_reduction_passes_despite_bits_lost() -> None:
    seq = "x = xs[0]\nacc = torch.zeros(x.shape[0])\nfor i in range(x.shape[1]):\n    acc += x[:, i]\nreturn acc"
    result = _run(REF_ROWSUM, _cand(seq), draws=1)
    assert result["verdict"] == "pass", result
    assert result["worst"]["bits_lost"] > 2.0


def test_onepass_variance_layernorm_passes() -> None:
    onepass = (
        "x = xs[0]\nmu = x.mean(-1, keepdim=True)\n"
        "var = (x * x).mean(-1, keepdim=True) - mu * mu\nreturn (x - mu) / torch.sqrt(var + 1e-5)"
    )
    assert _verdict(REF_LAYERNORM_MEAN50, _cand(onepass)) == "pass"
    half = "return F.layer_norm(xs[0].half(), xs[0].shape[-1:]).float()"
    assert _verdict(REF_LAYERNORM_MEAN50, _cand(half)) == "flag"


def test_plain_tensor_attribute_follows_reference_precision() -> None:
    honest = "x = xs[0]\nreturn torch.tanh(x @ x.T) @ self.h0"
    result = _run(REF_PLAIN_ATTR, _cand(honest, init="self.h0 = torch.randn(8, 64)"))
    assert result["verdict"] == "pass", result
    assert result["ref64"]["low_precision_ops"] == {}


def test_reference_below_fp64_is_inconclusive() -> None:
    result = _run(REF_BELOW_FP64, _cand("a, b = xs\nreturn a @ b"))
    assert result["verdict"] == "inconclusive"
    assert "mm" in result["ref64"]["low_precision_ops"]


def test_output_without_precision_signal_is_inconclusive() -> None:
    assert _verdict(REF_SIGN, _cand("return torch.sign(xs[0])")) == "inconclusive"


def test_nonfinite_where_reference_finite_flags() -> None:
    bad = "y = torch.sign(xs[0]).clone()\ny[0, 0] = float('inf')\nreturn y"
    result = _run(REF_SIGN, _cand(bad))
    assert result["verdict"] == "flag"
    assert result["worst"]["nonfinite_mismatch"] == 3


def test_overflow_shared_with_fp32_reference_not_charged() -> None:
    ref = REF_SIGN.replace("torch.sign(x)", "torch.exp(x * 30)").replace("32, 32", "64, 1024")
    result = _run(ref, _cand("return torch.exp(xs[0] * 30)"))
    assert result["verdict"] == "pass", result
    assert result["worst"]["nonfinite_mismatch"] == 0


def test_reference_crash_on_later_draw_classified() -> None:
    ref = REF_MATMUL.replace(
        "    def forward(self, a, b):\n",
        "    calls = 0\n\n    def forward(self, a, b):\n"
        "        Model.calls += 1\n        if Model.calls > 6:\n            raise RuntimeError('ref')\n",
    )
    result = _run(ref, _cand("a, b = xs\nreturn a @ b"))
    assert result["status"] == "ref_broken"
    assert result["draw"] == 1


def test_train_mode_only_downcast_needs_train_config() -> None:
    cand = _cand(
        "x = xs[0].half().float() if self.training else xs[0]\nreturn self.fc(x)", init=LINEAR_INIT
    )
    assert _verdict(REF_LINEAR, cand) == "pass"
    assert _verdict(REF_LINEAR, cand, train=True) == "flag"


def test_no_grad_only_downcast_needs_no_grad_config() -> None:
    cand = _cand(
        "x = xs[0] if torch.is_grad_enabled() else xs[0].half().float()\nreturn self.fc(x)",
        init=LINEAR_INIT,
    )
    assert _verdict(REF_LINEAR, cand) == "pass"
    assert _verdict(REF_LINEAR, cand, no_grad=True) == "flag"


def test_candidate_enabled_tf32_recorded_and_flags_restored() -> None:
    entry = _backend_flags()
    header = "torch.backends.cuda.matmul.allow_tf32 = True"
    result = _run(REF_MATMUL, _cand("a, b = xs\nreturn a @ b", header=header))
    assert result["backend_flags_after_cand_import"]["matmul.allow_tf32"] is True
    assert result["backend_flags_after_cand_import"]["cudnn.allow_tf32"] is False
    assert _backend_flags() == entry


@pytest.mark.parametrize(
    ("ref", "cand", "status"),
    [
        (REF_BROKEN, _cand("return xs[0]"), "ref_broken"),
        (REF_INT_ONLY, _cand("return xs[0] * 2"), "not_auditable"),
        (REF_MATMUL, "this is not python", "cand_load_fail"),
        (REF_MATMUL, _cand("raise RuntimeError('kernel launch failed')"), "cand_native_fail"),
        (REF_MATMUL, _cand("a, b = xs\nreturn (a @ b)[:-1]"), "cand_output_mismatch"),
    ],
    ids=["ref-broken", "no-float-input", "load-fail", "native-fail", "shape-mismatch"],
)
def test_status_classification(ref: str, cand: str, status: str) -> None:
    assert _run(ref, cand)["status"] == status
