"""Toy validation op (elementwise SiLU): gates + scoring bridge end-to-end."""

from __future__ import annotations

import torch
from torch import Tensor

from flash_mamba_rl.rl.prompts import available_ops, build_op_prompt
from flash_mamba_rl.verifier.op_harness import (
    elementwise_silu_reference,
    verify_elementwise_op,
)

EXCLUDED = ("gate_cmp_02_gradient_correctness",)


def correct_silu(x: Tensor) -> Tensor:
    if x.dtype in (torch.float32, torch.float64):
        return x * torch.sigmoid(x)
    x32 = x.to(torch.float32)
    return (x32 * torch.sigmoid(x32)).to(x.dtype)


def test_correct_eager_passes_all_gates() -> None:
    results = verify_elementwise_op(correct_silu, device="cpu")
    failed = {n: r.reason for n, r in results.items() if not r.passed}
    assert not failed, failed


def test_wrong_math_fails_value_gates() -> None:
    def wrong(x: Tensor) -> Tensor:
        return correct_silu(1.01 * x)

    results = verify_elementwise_op(wrong, device="cpu")
    assert not results["gate_cmp_01_input_variation"].passed


def test_nan_masking_cheat_fails_exc01() -> None:
    def masked(x: Tensor) -> Tensor:
        return torch.nan_to_num(correct_silu(x))

    results = verify_elementwise_op(masked, device="cpu")
    assert not results["gate_exc_01_exceptional_values"].passed


def test_nondeterminism_fails_ord02() -> None:
    state = {"calls": 0}

    def jitter(x: Tensor) -> Tensor:
        state["calls"] += 1
        return correct_silu(x) + (1e-3 if state["calls"] % 2 == 0 else 0.0)

    results = verify_elementwise_op(jitter, device="cpu")
    assert not results["gate_ord_02_atomic_determinism"].passed


def test_score_candidate_source_end_to_end() -> None:
    from flash_mamba_rl.verifier.candidate_scoring import score_candidate_source

    source = """
import torch

def elementwise_silu(x):
    if x.dtype in (torch.float32, torch.float64):
        return x * torch.sigmoid(x)
    x32 = x.to(torch.float32)
    return (x32 * torch.sigmoid(x32)).to(x.dtype)
"""
    result = score_candidate_source(source, op="elementwise_silu", device="cpu")
    assert result["status"] == "scored", result
    failed = {g: r for g, r in result["gates"].items() if not r["passed"]}
    assert result["contracts_passed"] is True, failed
    assert result["reward"] == 0.5


def test_prompt_registered() -> None:
    assert "elementwise_silu" in available_ops()
    prompt = build_op_prompt("elementwise_silu")
    assert "def elementwise_silu(" in prompt
    assert "sigmoid" in prompt


def test_reference_mixed_precision_rounds_once() -> None:
    x = torch.randn(4, 8, 16, dtype=torch.float16)
    out = elementwise_silu_reference(x)
    assert out.dtype == torch.float16
    expected = (x.float() * torch.sigmoid(x.float())).to(torch.float16)
    assert torch.equal(out, expected)
