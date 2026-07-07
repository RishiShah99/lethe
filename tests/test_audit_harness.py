"""Audit-harness behavior pins: planted-defect candidates and N/A reclassification.

Each source pair below follows the KernelBench convention the audited
corpora use (Model / ModelNew / get_inputs / get_init_inputs). Candidates
plant one defect each; the tests pin which gate class catches it and that
reference-side incapacity reclassifies to not-applicable instead of
inflating the failure stats.
"""

from __future__ import annotations

from typing import Any

from lethe.verifier.audit_harness import (
    _RI_MARKER,
    AUDIT_GATE_NAMES,
    GATE_SHORT_NAMES,
    _gate_status,
    _shape_variants,
    audit_worker,
)
from lethe.verifier.contracts import GateResult

REF_ELEMENTWISE = """
import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.abs(x) - 1.0


def get_inputs():
    return [torch.randn(8, 16)]


def get_init_inputs():
    return []
"""

CAND_HONEST = """
import torch


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.abs(x) - 1.0
"""

CAND_WRONG_VALUE = """
import torch


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.abs(x) * 1.01 - 1.0
"""

CAND_NONDETERMINISTIC = """
import torch


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.abs(x) - 1.0 + torch.rand_like(x) * 1e-3
"""

CAND_NAN_SWALLOW = """
import torch


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.nan_to_num(torch.abs(x) - 1.0, nan=0.0, posinf=0.0, neginf=0.0)
"""

CAND_BUFFER_REUSE = """
import torch


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._buf = {}

    def forward(self, x):
        key = (x.shape, x.dtype)
        y = self._buf.get(key)
        if y is None:
            y = torch.empty_like(x)
            self._buf[key] = y
        torch.sub(torch.abs(x), 1.0, out=y)
        return y
"""

CAND_SHAPE_RIGID = """
import torch


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert x.shape == (8, 16), "fixed-shape kernel"
        return torch.abs(x) - 1.0
"""

CAND_NATIVE_FAIL = """
import torch


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise RuntimeError("launch failed")
"""

CAND_SYNTAX_ERROR = """
class ModelNew(
"""

REF_SHAPE_RIGID = """
import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert x.shape == (8, 16), "task is shape-fixed"
        return torch.abs(x) - 1.0


def get_inputs():
    return [torch.randn(8, 16)]


def get_init_inputs():
    return []
"""

REF_LINEAR = """
import torch


class Model(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


def get_inputs():
    return [torch.randn(4, 32)]


def get_init_inputs():
    return [32, 8]
"""

CAND_LINEAR_HONEST = """
import torch


class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)
"""

CAND_FP16_ACCUM = """
import torch


class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        w = self.linear.weight.half().to(x.dtype)
        b = self.linear.bias.half().to(x.dtype)
        return (x.half() @ w.t().half() + b.half()).to(x.dtype)
"""

REF_BROKEN = """
import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise RuntimeError("reference is broken")


def get_inputs():
    return [torch.randn(4, 4)]


def get_init_inputs():
    return []
"""

REF_NO_TENSOR = """
import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, n):
        return n


def get_inputs():
    return [5]


def get_init_inputs():
    return []
"""


def _gates(result: dict[str, Any]) -> dict[str, str]:
    assert result["status"] == "gated", result
    return {k: v["status"] for k, v in result["gates"].items()}


def test_honest_candidate_passes_all_applicable_gates() -> None:
    result = audit_worker(REF_ELEMENTWISE, CAND_HONEST, {"device": "cpu"})
    gates = _gates(result)
    assert len(gates) == len(AUDIT_GATE_NAMES)
    assert all(s in ("pass", "na") for s in gates.values()), gates
    assert gates["CMP-01"] == "pass"
    assert gates["ORD-02"] == "pass"
    assert gates["EXC-01"] == "pass"
    assert not result["output_aliasing"]


def test_wrong_value_fails_cmp01() -> None:
    gates = _gates(audit_worker(REF_ELEMENTWISE, CAND_WRONG_VALUE, {"device": "cpu"}))
    assert gates["CMP-01"] == "fail"


def test_nondeterminism_fails_ord02() -> None:
    gates = _gates(audit_worker(REF_ELEMENTWISE, CAND_NONDETERMINISTIC, {"device": "cpu"}))
    assert gates["ORD-02"] == "fail"


def test_nan_swallowing_fails_exc01() -> None:
    gates = _gates(audit_worker(REF_ELEMENTWISE, CAND_NAN_SWALLOW, {"device": "cpu"}))
    assert gates["EXC-01"] == "fail"


def test_buffer_reuse_recorded_as_aliasing_and_ord02_still_meaningful() -> None:
    result = audit_worker(REF_ELEMENTWISE, CAND_BUFFER_REUSE, {"device": "cpu"})
    assert result["status"] == "gated"
    assert result["output_aliasing"] is True
    assert result["gates"]["ORD-02"]["status"] == "pass"
    assert result["gates"]["CMP-01"]["status"] == "pass"


def test_shape_rigid_candidate_fails_cmp03_against_flexible_reference() -> None:
    gates = _gates(audit_worker(REF_ELEMENTWISE, CAND_SHAPE_RIGID, {"device": "cpu"}))
    assert gates["CMP-03"] == "fail"
    assert gates["CMP-01"] == "fail"  # long_seq variant


def test_shape_rigid_reference_reclassifies_cmp03_to_na() -> None:
    gates = _gates(audit_worker(REF_SHAPE_RIGID, CAND_SHAPE_RIGID, {"device": "cpu"}))
    assert gates["CMP-03"] == "na"
    assert gates["ORD-02"] == "pass"


def test_shape_rigid_reference_cmp01_passes_with_skipped_long_seq() -> None:
    result = audit_worker(REF_SHAPE_RIGID, CAND_SHAPE_RIGID, {"device": "cpu"})
    cmp01 = result["gates"]["CMP-01"]
    assert cmp01["status"] == "pass"
    assert cmp01["skipped"] == 1


def test_linear_honest_passes_prc_gates() -> None:
    gates = _gates(audit_worker(REF_LINEAR, CAND_LINEAR_HONEST, {"device": "cpu"}))
    assert gates["PRC-01"] in ("pass", "na")
    assert gates["PRC-02"] in ("pass", "na")
    assert gates["CMP-01"] == "pass"


def test_fp16_accumulator_fails_a_precision_gate() -> None:
    gates = _gates(audit_worker(REF_LINEAR, CAND_FP16_ACCUM, {"device": "cpu"}))
    assert "fail" in (gates["PRC-01"], gates["PRC-02"], gates["CMP-01"]), gates


def test_candidate_native_failure_classified() -> None:
    result = audit_worker(REF_ELEMENTWISE, CAND_NATIVE_FAIL, {"device": "cpu"})
    assert result["status"] == "cand_native_fail"
    assert "launch failed" in result["error"]


def test_candidate_syntax_error_classified_as_load_fail() -> None:
    result = audit_worker(REF_ELEMENTWISE, CAND_SYNTAX_ERROR, {"device": "cpu"})
    assert result["status"] == "cand_load_fail"


def test_broken_reference_excluded() -> None:
    result = audit_worker(REF_BROKEN, CAND_HONEST, {"device": "cpu"})
    assert result["status"] == "ref_broken"


def test_no_tensor_input_not_auditable() -> None:
    result = audit_worker(REF_NO_TENSOR, CAND_HONEST, {"device": "cpu"})
    assert result["status"] == "not_auditable"


def test_shape_variants_exclude_native_and_dedupe() -> None:
    variants = _shape_variants((8, 16))
    assert (8, 16) not in variants
    assert (4, 16) in variants
    assert (16, 16) in variants
    assert (8, 32) in variants
    assert (8, 8) in variants
    assert len(variants) == len(set(variants))


def test_gate_short_names_cover_audit_set() -> None:
    assert set(GATE_SHORT_NAMES) >= set(AUDIT_GATE_NAMES)


REF_NAN_EMITTING = """
import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.log(x)


def get_inputs():
    return [torch.randn(8, 16)]


def get_init_inputs():
    return []
"""

CAND_NAN_MATCHING = """
import torch


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.log(x)
"""

CAND_UNDEEPCOPYABLE = """
import threading

import torch


class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self._lock = threading.Lock()

    def forward(self, x):
        return self.linear(x)
"""


def test_nan_matching_candidate_not_charged() -> None:
    # log(randn) emits NaN on negatives in BOTH implementations; positional
    # NaN agreement must not count as a value failure (C6.g-review FIX #2).
    gates = _gates(audit_worker(REF_NAN_EMITTING, CAND_NAN_MATCHING, {"device": "cpu"}))
    assert gates["CMP-01"] == "pass", gates
    assert gates["CMP-03"] == "pass", gates
    assert gates["EXC-01"] == "pass", gates


def test_undeepcopyable_candidate_prc_gates_via_rebuild() -> None:
    # Half-dtype variants rebuild by re-instantiation under the audit seed;
    # a deepcopy-hostile module must not be charged on the PRC gates
    # (C6.g-review FIX #3).
    gates = _gates(audit_worker(REF_LINEAR, CAND_UNDEEPCOPYABLE, {"device": "cpu"}))
    assert gates["PRC-01"] in ("pass", "na"), gates
    assert gates["PRC-02"] in ("pass", "na"), gates
    assert gates["CMP-01"] == "pass", gates


def test_ri_marker_not_forgeable_by_candidate_text() -> None:
    # H4: a candidate whose exception text merely contains the guessable
    # "[ref-inapplicable]" substring must NOT have its real failure
    # reclassified as skipped coverage — the marker carries a nonce only the
    # reference adapter produces. Only the genuine marker maps to "na".
    forged = GateResult(
        passed=False,
        reason="candidate raised: ValueError [ref-inapplicable] from foreign code",
        details={},
    )
    assert _gate_status("gate_cmp_01_input_variation", forged, None)["status"] == "fail"

    genuine = GateResult(passed=False, reason=f"{_RI_MARKER} ValueError: ref cannot", details={})
    assert _gate_status("gate_cmp_01_input_variation", genuine, None)["status"] == "na"
