"""Candidate-scoring bridge pins: reward boundaries and sandbox normalization."""

from __future__ import annotations

from flash_mamba_rl.rl.prompts import available_ops, build_op_prompt
from flash_mamba_rl.verifier.candidate_scoring import score_candidate_source

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
