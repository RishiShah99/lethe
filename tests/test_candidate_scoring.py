"""Candidate-scoring bridge pins: reward boundaries and sandbox normalization."""

from __future__ import annotations

from typing import Any

import pytest

from flash_mamba_rl.rl.prompts import available_ops, build_op_prompt
from flash_mamba_rl.rl.train import _OP_ENTRY_POINTS
from flash_mamba_rl.verifier import op_harness
from flash_mamba_rl.verifier.candidate_scoring import (
    _OP_VERIFIERS,
    OpSpec,
    _score_source_body,
    score_candidate_source,
    scoreable_ops,
)
from flash_mamba_rl.verifier.contracts import GateResult

# A correct eager candidate: mirrors the reference math exactly.
CORRECT_EAGER = """
import torch
import torch.nn.functional as F


def forward_chunked_scan(u, delta, A, B, C, D, *, chunk_size=64):
    work = u.dtype if u.dtype in (torch.float32, torch.float64) else torch.float32
    uw, dw = u.to(work), delta.to(work)
    Bw, Cw = B.to(work), C.to(work)
    delta_bar = F.softplus(dw)
    a_bar = torch.exp(delta_bar.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
    b_bar = delta_bar.unsqueeze(-1) * Bw.unsqueeze(2)
    batch, seq_len, d_model = uw.shape
    h = torch.zeros(batch, d_model, A.shape[1], dtype=work, device=u.device)
    y = torch.empty_like(uw)
    for t in range(seq_len):
        h = a_bar[:, t] * h + b_bar[:, t] * uw[:, t].unsqueeze(-1)
        y[:, t] = (h * Cw[:, t].unsqueeze(1)).sum(-1) + D * uw[:, t]
    return y.to(u.dtype)
"""

WRONG_MATH = CORRECT_EAGER.replace("+ D * uw[:, t]", "+ 2.0 * D * uw[:, t]")

SYNTAX_ERROR = "def forward_chunked_scan(u, delta"

NO_ENTRYPOINT = "import torch\n\ndef some_other_function(x):\n    return x\n"

HANGING = """
import time

def forward_chunked_scan(u, delta, A, B, C, D, *, chunk_size=64):
    time.sleep(600)
"""


def test_correct_eager_candidate_passes_contracts() -> None:
    result = score_candidate_source(CORRECT_EAGER, device="cpu")
    assert result["status"] == "scored", result
    assert result["compiled"] is True
    failed = {g: r for g, r in result["gates"].items() if not r["passed"]}
    assert result["contracts_passed"] is True, failed
    assert result["reward"] == 0.5  # contracts pass, no timing supplied


def test_wrong_math_scores_contract_failure() -> None:
    result = score_candidate_source(WRONG_MATH, device="cpu")
    assert result["status"] == "scored"
    assert result["contracts_passed"] is False
    assert result["reward"] == 0.1


def test_syntax_error_scores_zero() -> None:
    result = score_candidate_source(SYNTAX_ERROR, device="cpu")
    assert result["status"] == "exec_fail"
    assert result["compiled"] is False
    assert result["reward"] == 0.0


def test_missing_entrypoint_scores_zero() -> None:
    result = score_candidate_source(NO_ENTRYPOINT, device="cpu")
    assert result["status"] == "no_entrypoint"
    assert result["reward"] == 0.0


def test_hang_normalizes_to_sandbox_timeout() -> None:
    result = score_candidate_source(HANGING, device="cpu", timeout_s=10.0)
    assert result["status"] == "sandbox_timeout"
    assert result["reward"] == 0.0


def test_prompt_exists_for_scored_op() -> None:
    assert "forward_chunked_scan" in available_ops()
    prompt = build_op_prompt("forward_chunked_scan")
    assert "def forward_chunked_scan(" in prompt
    assert "softplus" in prompt
    assert "atomics" in prompt


def test_curriculum_registry_consistent() -> None:
    """Every scoreable op has a prompt, an entry point, and a real verify driver."""
    for op in scoreable_ops():
        assert op in available_ops()
        assert op in _OP_ENTRY_POINTS
        spec = _OP_VERIFIERS[op]
        assert _OP_ENTRY_POINTS[op] == spec.entry_point
        assert f"def {spec.entry_point}(" in build_op_prompt(op)
        assert callable(getattr(op_harness, spec.verify_name))
        if spec.view_fields_attr is not None:
            fields = getattr(op_harness, spec.view_fields_attr)
            assert len(fields) > 1


def test_forbidden_import_rejected() -> None:
    wrapper = (
        "from flash_mamba_rl.kernels.references.forward_chunked_scan "
        "import reference_forward_chunked_scan as forward_chunked_scan\n"
    )
    result = _score_source_body(wrapper, {"op": "forward_chunked_scan"})
    assert result["status"] == "forbidden_import"
    assert result["reward"] == 0.0
    dynamic = "import importlib\nforward_chunked_scan = None\n"
    assert _score_source_body(dynamic, {})["status"] == "forbidden_import"
    sys_fishing = "import sys as s\nforward_chunked_scan = None\n"
    assert _score_source_body(sys_fishing, {})["status"] == "forbidden_import"
    modules_fishing = 'm = __builtins__\nx = "sys.modules"\nforward_chunked_scan = None\n'
    assert _score_source_body(modules_fishing, {})["status"] == "forbidden_import"


# The exact reward-hack the AST screen + import guard exist to kill: build the
# oracle import name from split strings so a substring screen never sees it,
# reach it through __builtins__, and bind the reference as the entry point.
REFERENCE_WRAP_EXPLOIT = (
    'forward_chunked_scan = __builtins__["__im" + "port__"](\n'
    '    "flash" + "_mamba_rl.kernels.references.forward_chunked_scan",\n'
    '    fromlist=["reference_forward_chunked_scan"],\n'
    ").reference_forward_chunked_scan\n"
)


def test_reference_wrap_via_builtins_is_rejected() -> None:
    result = _score_source_body(REFERENCE_WRAP_EXPLOIT, {"op": "forward_chunked_scan"})
    assert result["status"] == "forbidden_import", result
    assert result["reward"] == 0.0


def test_import_guard_blocks_oracle_reimport() -> None:
    """Belt-and-suspenders: a runtime-built oracle name still resolves through
    the guarded __import__ and is rejected, while honest imports pass —
    importlib INCLUDED, because a real Triton candidate pulls it in transitively
    at jit time and guard-blocking it scored every kernel as a non-compile."""
    import builtins
    import importlib
    import math

    from flash_mamba_rl.verifier.candidate_scoring import _import_guard

    real = builtins.__import__
    with _import_guard():
        with pytest.raises(ImportError):
            builtins.__import__("flash_mamba_rl.kernels.references.forward_chunked_scan")
        assert builtins.__import__("importlib") is importlib  # NOT blocked
        assert builtins.__import__("math") is math
    assert builtins.__import__ is real  # restored on exit


def test_oracle_import_blocker_covers_importlib_pathway() -> None:
    """The meta-path finder blocks the oracle through importlib.import_module —
    which bypasses builtins.__import__ — while non-oracle imports fall through.
    With the oracle evicted from sys.modules, the re-import re-consults it."""
    import importlib

    from flash_mamba_rl.verifier.candidate_scoring import (
        _hidden_project_modules,
        _oracle_import_blocker,
    )

    with _hidden_project_modules(), _oracle_import_blocker():
        with pytest.raises(ImportError):
            importlib.import_module("flash_mamba_rl.kernels.references.forward_chunked_scan")
        assert importlib.import_module("math").pi  # non-oracle passes through
    assert importlib.import_module("flash_mamba_rl.verifier.reward") is not None  # removed on exit


# Bypasses the AST screen (no forbidden Name nodes — __builtins__/__import__/
# importlib appear only as string constants and attribute accesses), reaches a
# live __import__ through globals()['__builtins__'], imports importlib (now
# allowed past the guard), and tries importlib.import_module on the oracle. Only
# the meta-path finder stops it — the test that the finder is load-bearing.
ORACLE_GADGET_VIA_FINDER = (
    "b = globals()['__builtins__']\n"
    "imp = b['__import__'] if isinstance(b, dict) else b.__import__\n"
    "il = imp('importlib')\n"
    "forward_chunked_scan = il.import_module(\n"
    "    'flash' + '_mamba_rl.kernels.references.forward_chunked_scan'\n"
    ").reference_forward_chunked_scan\n"
)


def test_oracle_gadget_through_importlib_is_blocked_by_finder() -> None:
    result = _score_source_body(ORACLE_GADGET_VIA_FINDER, {"op": "forward_chunked_scan"})
    assert result["status"] == "exec_fail", result
    assert result["reward"] == 0.0
    assert "may not reach the oracle" in result["error"]


def test_ast_screen_passes_clean_candidate() -> None:
    from flash_mamba_rl.verifier.candidate_scoring import _ast_screen

    assert _ast_screen(CORRECT_EAGER) == []
    # A syntactically broken source is not a screen hit — exec classifies it.
    assert _ast_screen(SYNTAX_ERROR) == []


def test_project_modules_hidden_during_candidate_exec() -> None:
    import sys

    from flash_mamba_rl.verifier.candidate_scoring import _hidden_project_modules

    before = {k: v for k, v in sys.modules.items() if k.startswith("flash_mamba_rl")}
    assert before  # the package is imported in this test process
    with _hidden_project_modules():
        assert not any(k.startswith("flash_mamba_rl") for k in sys.modules)
    after = {k: v for k, v in sys.modules.items() if k.startswith("flash_mamba_rl")}
    assert after == before  # identical module objects restored


# ---------------------------------------------------------------------------
# Multi-view aggregation (fake driver, in-process)
# ---------------------------------------------------------------------------

FAKE_FIELDS = ("grad_p", "grad_q", "grad_r")

FAKE_SOURCE = "def fake_bwd(dy):\n    return dy\n"

_PASS = GateResult(passed=True, reason="")
_FAIL = GateResult(passed=False, reason="boom")


def _fake_battery(ok: bool) -> dict[str, GateResult]:
    return {
        "gate_cmp_01_input_variation": _PASS if ok else _FAIL,
        "gate_cmp_02_gradient_correctness": _FAIL,  # excluded by default
        "gate_ord_02_determinism": _PASS,
    }


@pytest.fixture()
def fake_bwd_op(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    calls: dict[str, Any] = {"fields": [], "passing": set(FAKE_FIELDS)}

    def verify_fake(fn: Any, *, grad_field: str, device: str = "cpu") -> dict[str, GateResult]:
        calls["fields"].append(grad_field)
        return _fake_battery(grad_field in calls["passing"])

    monkeypatch.setattr(op_harness, "FAKE_FIELDS", FAKE_FIELDS, raising=False)
    monkeypatch.setattr(op_harness, "verify_fake", verify_fake, raising=False)
    monkeypatch.setitem(
        _OP_VERIFIERS, "fake_bwd_op", OpSpec("fake_bwd", "verify_fake", "FAKE_FIELDS")
    )
    return calls


class TestMultiView:
    def test_all_views_pass_is_contract_pass(self, fake_bwd_op: dict[str, Any]) -> None:
        result = _score_source_body(FAKE_SOURCE, {"op": "fake_bwd_op"})
        assert result["contracts_passed"] is True
        assert result["views_passed"] == 3
        assert result["views_total"] == 3
        assert result["reward"] == 0.5
        assert "grad_p/gate_cmp_01_input_variation" in result["gates"]

    def test_one_failing_view_fails_contract(self, fake_bwd_op: dict[str, Any]) -> None:
        fake_bwd_op["passing"] = {"grad_p", "grad_r"}
        result = _score_source_body(FAKE_SOURCE, {"op": "fake_bwd_op", "fail_fast": False})
        assert result["contracts_passed"] is False
        assert result["views_passed"] == 2
        assert result["first_failed_view"] == "grad_q"
        assert result["reward"] == 0.1

    def test_fail_fast_stops_at_first_failing_view(self, fake_bwd_op: dict[str, Any]) -> None:
        fake_bwd_op["passing"] = set()
        result = _score_source_body(FAKE_SOURCE, {"op": "fake_bwd_op", "fail_fast": True})
        assert fake_bwd_op["fields"] == ["grad_p"]
        assert result["views_passed"] == 0
        assert result["first_failed_view"] == "grad_p"

    def test_view_fraction_shaping_stays_below_full_pass(self, fake_bwd_op: dict[str, Any]) -> None:
        fake_bwd_op["passing"] = {"grad_p", "grad_q"}
        result = _score_source_body(
            FAKE_SOURCE,
            {"op": "fake_bwd_op", "fail_fast": False, "reward_shaping": "view_fraction"},
        )
        assert result["reward"] == pytest.approx(0.1 + 0.35 * 2 / 3)
        assert result["reward"] < 0.5
        fake_bwd_op["passing"] = set(FAKE_FIELDS)
        full = _score_source_body(
            FAKE_SOURCE, {"op": "fake_bwd_op", "reward_shaping": "view_fraction"}
        )
        assert full["reward"] == 0.5

    def test_excluded_gate_failure_does_not_charge_view(self, fake_bwd_op: dict[str, Any]) -> None:
        # CMP-02 fails in every fake battery; default exclusion keeps views green.
        result = _score_source_body(FAKE_SOURCE, {"op": "fake_bwd_op"})
        assert result["contracts_passed"] is True

    def test_shaping_forces_full_battery(self, fake_bwd_op: dict[str, Any]) -> None:
        """view_fraction must count passes, not a fail-fast prefix."""
        fake_bwd_op["passing"] = {"grad_q", "grad_r"}  # first view fails
        result = _score_source_body(
            FAKE_SOURCE,
            {"op": "fake_bwd_op", "fail_fast": True, "reward_shaping": "view_fraction"},
        )
        assert fake_bwd_op["fields"] == list(FAKE_FIELDS)  # no early stop
        assert result["views_passed"] == 2
        assert result["reward"] == pytest.approx(0.1 + 0.35 * 2 / 3)
