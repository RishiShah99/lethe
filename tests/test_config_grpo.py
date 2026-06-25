"""CPU tests for the E2.c config-emitting GRPO path.

Covers the action representation (extract/parse), the parse-then-score bridge
on CPU (where a config runs the eager path -> the 0.5 correct-but-slow floor,
no speedup measured), the config-emission prompt, and the GRPOTrainingLoop
reuse through its extractor + scorer hooks with a stub policy (so the full
advantages -> loss -> update path runs and parameter movement is observable).
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest
import torch

from flash_mamba_rl.kernels.autotune import KernelConfig, ShapeSpec, validate
from flash_mamba_rl.rl.config_grpo import (
    build_config_scorer,
    extract_config,
    parse_config,
    score_config_candidate,
    serial_seed_completions,
)
from flash_mamba_rl.rl.prompts import build_config_prompt
from flash_mamba_rl.rl.train import GRPOTrainingLoop, TrainLoopConfig


class TestExtractConfig:
    def test_picks_fenced_json_block(self) -> None:
        text = 'reasoning...\n```json\n{"num_warps": 4}\n```'
        assert extract_config(text) == '{"num_warps": 4}'

    def test_picks_last_block(self) -> None:
        text = '```json\n{"a": 1}\n```\nthen\n```json\n{"num_warps": 8}\n```'
        assert extract_config(text) == '{"num_warps": 8}'

    def test_falls_back_to_bare_body(self) -> None:
        assert extract_config('  {"num_warps": 2}  ') == '{"num_warps": 2}'

    def test_empty_returns_none(self) -> None:
        assert extract_config("   \n  ") is None


class TestParseConfig:
    def test_valid(self) -> None:
        assert parse_config('{"num_warps": 4, "block_d": 32}') == KernelConfig(
            num_warps=4, block_d=32
        )

    def test_empty_object_is_default(self) -> None:
        assert parse_config("{}") == KernelConfig()

    def test_unknown_knob_rejected(self) -> None:
        assert parse_config('{"block_z": 4}') is None

    def test_bool_value_rejected(self) -> None:
        # bool is an int subclass; a JSON true is not a launch knob.
        assert parse_config('{"num_warps": true}') is None

    def test_float_value_rejected(self) -> None:
        assert parse_config('{"num_warps": 4.0}') is None

    def test_non_dict_rejected(self) -> None:
        assert parse_config("[1, 2, 3]") is None
        assert parse_config("4") is None

    def test_bad_json_rejected(self) -> None:
        assert parse_config("{not json}") is None

    def test_out_of_grid_value_still_parses(self) -> None:
        # parse_config enforces only JSON shape; grid legality is validate's job.
        assert parse_config('{"num_warps": 3}') == KernelConfig(num_warps=3)

    def test_scan_mode_string_accepted(self) -> None:
        # scan_mode is the one string-valued knob; chunk_len stays an int.
        assert parse_config('{"scan_mode": "chunk_parallel", "chunk_len": 256}') == KernelConfig(
            scan_mode="chunk_parallel", chunk_len=256
        )

    def test_scan_mode_non_string_rejected(self) -> None:
        assert parse_config('{"scan_mode": 1}') is None

    def test_int_knob_as_string_still_rejected(self) -> None:
        # Only scan_mode may be a string; a stringified int knob is illegal.
        assert parse_config('{"num_warps": "4"}') is None


class TestScoreConfigCandidateCPU:
    def test_valid_config_passes_contracts(self) -> None:
        res = score_config_candidate(
            '{"num_warps": 8, "block_d": 64}',
            op="forward_chunked_scan",
            device="cpu",
            measure_speedup=False,
        )
        assert res["status"] == "scored"
        assert res["contracts_passed"] is True
        assert res["reward"] == 0.5

    def test_illegal_emission_scores_zero(self) -> None:
        res = score_config_candidate(
            "I think num_warps should be high",
            op="forward_chunked_scan",
            device="cpu",
        )
        assert res["status"] == "illegal_config"
        assert res["reward"] == 0.0
        assert res["contracts_passed"] is False

    def test_out_of_grid_demoted_to_invalid(self) -> None:
        res = score_config_candidate('{"num_warps": 3}', op="forward_chunked_scan", device="cpu")
        assert res["status"] == "invalid_config"
        assert res["reward"] == 0.0

    def test_build_config_scorer_closure(self) -> None:
        scorer = build_config_scorer(
            "forward_chunked_scan",
            ShapeSpec(2, 2048, 1024),
            device="cpu",
            measure_speedup=False,
        )
        res = scorer("{}")
        assert res["status"] == "scored"
        assert res["contracts_passed"] is True


class TestConfigPrompt:
    def test_states_shape_and_legal_knobs(self) -> None:
        prompt = build_config_prompt("fused_block_forward", ShapeSpec(2, 16384, 2048))
        assert "seq_len = 16384" in prompt
        assert "d_model = 2048" in prompt
        assert "block_d" in prompt and "num_warps" in prompt and "num_stages" in prompt
        # block_p is not tunable for this op -> must not be offered.
        assert "block_p" not in prompt

    def test_backward_op_mentions_chunk_k_divisibility(self) -> None:
        prompt = build_config_prompt("mimo_backward", ShapeSpec(2, 8192, 2048))
        assert "chunk_k" in prompt and "block_p" in prompt
        assert "divide seq_len" in prompt


# --- GRPOTrainingLoop reuse with a stub policy (no model, no GPU) -----------

GOOD = '```json\n{"num_warps": 4, "block_d": 32}\n```'
BAD = '```json\n{"num_warps": 8, "block_d": 16}\n```'


class StubTrainablePolicy:
    """Minimal TrainablePolicy: log-probs ride a trainable parameter."""

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


def _config_reward_scorer(text: str) -> dict[str, Any]:
    fast = '"num_warps": 4' in text
    return {
        "status": "scored",
        "reward": 1.0 if fast else 0.1,
        "compiled": True,
        "contracts_passed": True,
        "gates": {},
    }


class TestLoopReuseWithConfigExtractor:
    def test_extractor_feeds_json_to_scorer(self, tmp_path: Any) -> None:
        seen: list[str] = []

        def spy(text: str) -> dict[str, Any]:
            seen.append(text)
            return _config_reward_scorer(text)

        policy = StubTrainablePolicy([GOOD])
        config = TrainLoopConfig(
            op="fused_block_forward",
            n_per_prompt=4,
            total_steps=1,
            checkpoint_dir=str(tmp_path / "ckpt"),
            device="cpu",
        )
        loop = GRPOTrainingLoop(config, policy, prompt="P", extractor=extract_config, scorer=spy)
        loop.step()
        assert len(seen) == 4
        # The extractor pulled the fenced JSON, not the whole completion.
        assert all(s == '{"num_warps": 4, "block_d": 32}' for s in seen)

    def test_mixed_config_rewards_update_parameters(self, tmp_path: Any) -> None:
        policy = StubTrainablePolicy([GOOD, BAD])
        config = TrainLoopConfig(
            op="fused_block_forward",
            n_per_prompt=4,
            total_steps=1,
            checkpoint_dir=str(tmp_path / "ckpt"),
            device="cpu",
        )
        loop = GRPOTrainingLoop(
            config, policy, prompt="P", extractor=extract_config, scorer=_config_reward_scorer
        )
        before = policy.theta.detach().clone()
        metrics = loop.step()
        assert metrics.loss is not None
        assert metrics.mean_reward == pytest.approx(0.55)
        assert not torch.equal(policy.theta.detach(), before)


# --- #14 forced serial-seeding ---------------------------------------------

CHUNK_PARALLEL = '```json\n{"scan_mode": "chunk_parallel", "chunk_len": 256}\n```'


def _serial_pref_scorer(text: str) -> dict[str, Any]:
    """A scorer that rewards serial over chunk_parallel (the saturated regime)."""
    serial = '"serial"' in text
    return {
        "status": "scored",
        "reward": 1.0 if serial else 0.1,
        "compiled": True,
        "contracts_passed": True,
        "gates": {},
    }


class TestSerialSeedCompletions:
    def test_scan_mode_op_yields_grid_legal_serial_configs(self) -> None:
        seeds = serial_seed_completions("backward_selective_scan", 3)
        assert len(seeds) == 3
        cfgs = [parse_config(extract_config(s) or "") for s in seeds]
        assert all(c is not None and c.scan_mode == "serial" for c in cfgs)
        # Each seed pins a distinct grid num_warps; all are grid-legal.
        assert {c.num_warps for c in cfgs if c is not None} == {2, 4, 8}
        assert all(validate("backward_selective_scan", c) == [] for c in cfgs if c is not None)

    def test_no_scan_mode_op_yields_nothing(self) -> None:
        for op in ("mimo_backward", "complex_scan_rope", "fused_block_forward"):
            assert serial_seed_completions(op, 3) == []

    def test_zero_or_negative_yields_nothing(self) -> None:
        assert serial_seed_completions("forward_chunked_scan", 0) == []
        assert serial_seed_completions("forward_chunked_scan", -1) == []

    def test_warps_cycle_when_n_exceeds_grid(self) -> None:
        seeds = serial_seed_completions("forward_chunked_scan", 4)
        warps = [parse_config(extract_config(s) or "").num_warps for s in seeds]  # type: ignore[union-attr]
        assert warps == [2, 4, 8, 2]


class TestSerialSeedingInLoop:
    def test_unseeded_single_mode_group_is_degenerate(self, tmp_path: Any) -> None:
        # The #14 failure: a chunk_parallel-only group has one reward value, so
        # the update skips (loss=None) and no serial gradient ever forms.
        policy = StubTrainablePolicy([CHUNK_PARALLEL])
        config = TrainLoopConfig(
            op="forward_chunked_scan",
            n_per_prompt=6,
            total_steps=1,
            checkpoint_dir=str(tmp_path / "ckpt"),
            device="cpu",
        )
        loop = GRPOTrainingLoop(
            config, policy, prompt="P", extractor=extract_config, scorer=_serial_pref_scorer
        )
        assert loop.step().loss is None

    def test_seeds_break_degeneracy_and_record_serial(self, tmp_path: Any) -> None:
        seeds = serial_seed_completions("forward_chunked_scan", 2)
        policy = StubTrainablePolicy([CHUNK_PARALLEL])
        ckpt = tmp_path / "ckpt"
        config = TrainLoopConfig(
            op="forward_chunked_scan",
            n_per_prompt=6,
            total_steps=1,
            checkpoint_dir=str(ckpt),
            device="cpu",
        )
        before = policy.theta.detach().clone()
        loop = GRPOTrainingLoop(
            config,
            policy,
            prompt="P",
            extractor=extract_config,
            scorer=_serial_pref_scorer,
            seed_completions=seeds,
        )
        metrics = loop.step()
        # Seeds make the group non-degenerate -> a real update happens.
        assert metrics.loss is not None
        assert not torch.equal(policy.theta.detach(), before)

        rows = [json.loads(line) for line in (ckpt / "rollouts.jsonl").read_text().splitlines()]
        seeded = [r for r in rows if r["seeded"]]
        assert len(seeded) == 2
        assert all(r["idx"] in (0, 1) for r in seeded)
        assert all(r["reward"] == 1.0 for r in seeded)
        assert all('"serial"' in r["source"] for r in seeded)
        rest = [r for r in rows if not r["seeded"]]
        assert len(rest) == 4
        assert all(r["reward"] == 0.1 for r in rest)
