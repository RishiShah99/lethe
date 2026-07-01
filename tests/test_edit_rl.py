"""CPU tests for the E2.f edit-emitting GRPO path.

Covers the edit action representation (extract/parse/apply), the
apply-then-score bridge on CPU against the eager base (where the gate battery
re-verifies correctness — a no-op edit lands the 0.5 floor, a correctness-
breaking edit is demoted), the edit-emission prompt, the adapted
``score_candidate_source`` shape threading, the parallel scorer, and the
GRPOTrainingLoop reuse through its extractor + scorer hooks with a stub policy.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest
import torch

from flash_mamba_rl.kernels.autotune import ShapeSpec
from flash_mamba_rl.rl.edit_rl import (
    apply_edits,
    build_edit_scorer,
    extract_edits,
    parse_edits,
    score_edit_candidate,
)
from flash_mamba_rl.rl.parallel_scoring import ParallelEditScorer
from flash_mamba_rl.rl.prompts import build_edit_prompt
from flash_mamba_rl.rl.train import GRPOTrainingLoop, TrainLoopConfig
from flash_mamba_rl.verifier.candidate_scoring import score_candidate_source


def _block(search: str, replace: str) -> str:
    return f"<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE"


# Exact lines from the eager forward_chunked_scan SFT target.
NOOP_EDIT = _block("    out_dtype = u.dtype", "    out_dtype = u.dtype  # tuned")
BREAK_EDIT = _block(
    "        y[:, t, :] = (h * C[:, t, :].unsqueeze(1)).sum(-1) + D * u[:, t, :]",
    "        y[:, t, :] = (h * C[:, t, :].unsqueeze(1)).sum(-1)",
)
NOMATCH_EDIT = _block("not present anywhere in the base", "whatever")


class TestExtractEdits:
    def test_keeps_text_with_marker(self) -> None:
        assert extract_edits(f"some reasoning\n{NOOP_EDIT}") is not None

    def test_none_without_marker(self) -> None:
        assert extract_edits("num_warps should be 8, no edits") is None


class TestParseEdits:
    def test_single_block(self) -> None:
        assert parse_edits(NOOP_EDIT) == [
            ("    out_dtype = u.dtype", "    out_dtype = u.dtype  # tuned")
        ]

    def test_multiple_blocks_in_order(self) -> None:
        text = _block("a", "b") + "\nfiller\n" + _block("c", "d")
        assert parse_edits(text) == [("a", "b"), ("c", "d")]

    def test_no_block_is_none(self) -> None:
        assert parse_edits("just prose, no markers") is None

    def test_unterminated_block_is_none(self) -> None:
        # SEARCH opened, divider present, but REPLACE marker missing.
        assert parse_edits("<<<<<<< SEARCH\nx\n=======\ny\n") is None

    def test_missing_divider_is_none(self) -> None:
        assert parse_edits("<<<<<<< SEARCH\nx\n>>>>>>> REPLACE\n") is None

    def test_bare_divider_in_body_is_rejected_not_misparsed(self) -> None:
        # A body line that is itself a bare `=======` marker would shift the
        # block boundaries; the parser must reject the whole emission rather
        # than mis-parse it into a wrong edit.
        text = "<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n=======\n>>>>>>> REPLACE"
        assert parse_edits(text) is None

    def test_replace_marker_in_search_body_is_rejected(self) -> None:
        text = "<<<<<<< SEARCH\nfoo\n>>>>>>> REPLACE\n=======\nbar\n>>>>>>> REPLACE"
        assert parse_edits(text) is None


class TestApplyEdits:
    BASE = "alpha\nbeta\ngamma\n"

    def test_applies_unique_match(self) -> None:
        assert apply_edits(self.BASE, [("beta", "BETA")]) == "alpha\nBETA\ngamma\n"

    def test_sequential_blocks_see_prior_result(self) -> None:
        out = apply_edits(self.BASE, [("beta", "delta"), ("delta", "epsilon")])
        assert out == "alpha\nepsilon\ngamma\n"

    def test_absent_search_is_none(self) -> None:
        assert apply_edits(self.BASE, [("zeta", "x")]) is None

    def test_ambiguous_search_is_none(self) -> None:
        assert apply_edits("x\nx\n", [("x", "y")]) is None

    def test_empty_search_is_none(self) -> None:
        assert apply_edits(self.BASE, [("", "x")]) is None


class TestScoreEditCandidateCPU:
    def test_noop_edit_preserves_contracts(self) -> None:
        res = score_edit_candidate(
            NOOP_EDIT,
            op="forward_chunked_scan",
            base_variant="eager",
            device="cpu",
            measure_speedup=False,
        )
        assert res["status"] == "scored"
        assert res["contracts_passed"] is True
        assert res["reward"] == 0.5

    def test_correctness_breaking_edit_demoted(self) -> None:
        res = score_edit_candidate(
            BREAK_EDIT,
            op="forward_chunked_scan",
            base_variant="eager",
            device="cpu",
            measure_speedup=False,
        )
        assert res["status"] == "scored"
        assert res["contracts_passed"] is False
        assert res["reward"] < 0.5

    def test_no_block_is_illegal(self) -> None:
        res = score_edit_candidate(
            "I would increase the block size",
            op="forward_chunked_scan",
            base_variant="eager",
            device="cpu",
        )
        assert res["status"] == "illegal_edit"
        assert res["reward"] == 0.0

    def test_nonmatching_search_is_illegal(self) -> None:
        res = score_edit_candidate(
            NOMATCH_EDIT,
            op="forward_chunked_scan",
            base_variant="eager",
            device="cpu",
        )
        assert res["status"] == "illegal_edit"
        assert res["reward"] == 0.0

    def test_build_edit_scorer_closure(self) -> None:
        scorer = build_edit_scorer(
            "forward_chunked_scan",
            ShapeSpec(2, 64, 32),
            base_variant="eager",
            device="cpu",
            measure_speedup=False,
        )
        res = scorer(NOOP_EDIT)
        assert res["status"] == "scored" and res["contracts_passed"] is True


class TestEditPrompt:
    def test_embeds_shape_source_and_format(self) -> None:
        prompt = build_edit_prompt(
            "forward_chunked_scan", ShapeSpec(2, 16384, 1024), base_variant="eager"
        )
        assert "seq_len = 16384" in prompt and "d_model = 1024" in prompt
        # The base source is embedded so the policy can copy exact SEARCH text.
        assert "def forward_chunked_scan(" in prompt
        assert "<<<<<<< SEARCH" in prompt and ">>>>>>> REPLACE" in prompt


class TestScoreCandidateSourceShape:
    def test_shape_arg_does_not_break_cpu_scoring(self) -> None:
        from flash_mamba_rl.rl.sft_targets import target_source

        base = target_source("forward_chunked_scan", "eager")
        res = score_candidate_source(
            base, op="forward_chunked_scan", device="cpu", shape=(2, 4096, 1024)
        )
        assert res["status"] == "scored" and res["reward"] == 0.5


class TestParallelEditScorerCPU:
    def test_scores_batch_on_cpu(self) -> None:
        scorer = ParallelEditScorer(
            op="forward_chunked_scan",
            gpu_ids=(0,),
            shape=ShapeSpec(2, 64, 32),
            base_variant="eager",
            device="cpu",
            measure_speedup=False,
        )
        results = scorer([NOOP_EDIT, NOMATCH_EDIT])
        assert results[0]["contracts_passed"] is True and results[0]["reward"] == 0.5
        assert results[1]["status"] == "illegal_edit" and results[1]["reward"] == 0.0


# --- GRPOTrainingLoop reuse with a stub policy (no model, no GPU) -----------

FAST_EDIT = _block("    out_dtype = u.dtype", "    out_dtype = u.dtype  # fast")
SLOW_EDIT = _block("    out_dtype = u.dtype", "    out_dtype = u.dtype  # slow")


class StubTrainablePolicy:
    def __init__(self, completions: list[str]) -> None:
        self.theta = torch.nn.Parameter(torch.zeros(4))
        self._completions = completions
        self.last_terminated: list[bool] = []

    def generate(self, prompt: str, n: int) -> list[str]:
        out = [self._completions[i % len(self._completions)] for i in range(n)]
        self.last_terminated = [True] * n
        return out

    def completion_log_probs(
        self,
        prompt: str,
        completions: list[str],
        *,
        use_adapter: bool = True,
        append_eos: Any = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k, t = len(completions), self.theta.shape[0]
        base = -torch.nn.functional.softplus(self.theta).expand(k, t)
        if not use_adapter:
            base = base.detach() - 0.01
        mask = torch.ones(k, t, dtype=torch.bool)
        return base * mask, mask

    def trainable_parameters(self) -> Any:
        return iter([self.theta])

    def save_adapter(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

    def eval_mode(self) -> None:
        pass


def _edit_reward_scorer(text: str) -> dict[str, Any]:
    fast = "# fast" in text
    return {
        "status": "scored",
        "reward": 1.0 if fast else 0.1,
        "compiled": True,
        "contracts_passed": True,
        "gates": {},
    }


class TestLoopReuseWithEditExtractor:
    def test_extractor_feeds_edit_text_to_scorer(self, tmp_path: Any) -> None:
        seen: list[str] = []

        def spy(text: str) -> dict[str, Any]:
            seen.append(text)
            return _edit_reward_scorer(text)

        policy = StubTrainablePolicy([FAST_EDIT])
        config = TrainLoopConfig(
            op="forward_chunked_scan",
            n_per_prompt=4,
            total_steps=1,
            checkpoint_dir=str(tmp_path / "ckpt"),
            device="cpu",
        )
        loop = GRPOTrainingLoop(config, policy, prompt="P", extractor=extract_edits, scorer=spy)
        loop.step()
        assert len(seen) == 4
        assert all("<<<<<<< SEARCH" in s for s in seen)

    def test_mixed_edit_rewards_update_parameters(self, tmp_path: Any) -> None:
        policy = StubTrainablePolicy([FAST_EDIT, SLOW_EDIT])
        config = TrainLoopConfig(
            op="forward_chunked_scan",
            n_per_prompt=4,
            total_steps=1,
            checkpoint_dir=str(tmp_path / "ckpt"),
            device="cpu",
        )
        loop = GRPOTrainingLoop(
            config, policy, prompt="P", extractor=extract_edits, scorer=_edit_reward_scorer
        )
        before = policy.theta.detach().clone()
        metrics = loop.step()
        assert metrics.loss is not None
        assert metrics.mean_reward == pytest.approx(0.55)
        assert not torch.equal(policy.theta.detach(), before)

    def test_no_edit_completion_is_no_code_block(self, tmp_path: Any) -> None:
        policy = StubTrainablePolicy(["just prose, no edits"])
        config = TrainLoopConfig(
            op="forward_chunked_scan",
            n_per_prompt=3,
            total_steps=1,
            checkpoint_dir=str(tmp_path / "ckpt"),
            device="cpu",
        )
        loop = GRPOTrainingLoop(
            config, policy, prompt="P", extractor=extract_edits, scorer=_edit_reward_scorer
        )
        metrics = loop.step()
        assert metrics.n_no_code == 3
        rows = [
            json.loads(line)
            for line in (tmp_path / "ckpt" / "rollouts.jsonl").read_text().splitlines()
        ]
        assert all(r["status"] == "no_code_block" for r in rows)
