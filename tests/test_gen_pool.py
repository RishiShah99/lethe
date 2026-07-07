"""GenerationPool pins: even split, order preservation, refresh, loop wiring."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest
import torch

from lethe.rl.gen_pool import GenerationPool, split_counts


class StubReplica:
    """Records its adapter state and tags completions with its replica id."""

    def __init__(self, rid: int) -> None:
        self.rid = rid
        self.last_terminated: list[bool] = []
        self.loaded_state: dict[str, torch.Tensor] | None = None
        self.concurrent_peak = 0

    def generate(self, prompt: str, n: int) -> list[str]:
        out = [f"r{self.rid}_c{i}" for i in range(n)]
        # Even-completion = naturally terminated, odd = truncated.
        self.last_terminated = [i % 2 == 0 for i in range(n)]
        return out

    def load_adapter_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.loaded_state = state

    def eval_mode(self) -> None:
        pass


class TestSplitCounts:
    def test_even(self) -> None:
        assert split_counts(8, 4) == [2, 2, 2, 2]

    def test_uneven_front_loaded(self) -> None:
        assert split_counts(8, 3) == [3, 3, 2]
        assert split_counts(5, 4) == [2, 1, 1, 1]

    def test_more_parts_than_items(self) -> None:
        assert split_counts(2, 4) == [1, 1, 0, 0]

    def test_sum_preserved(self) -> None:
        for n in (1, 7, 8, 31):
            for parts in (1, 2, 3, 4, 8):
                assert sum(split_counts(n, parts)) == n

    def test_zero_parts_rejected(self) -> None:
        with pytest.raises(ValueError):
            split_counts(8, 0)


class TestGenerationPool:
    def test_requires_replicas(self) -> None:
        with pytest.raises(ValueError):
            GenerationPool([])

    def test_order_preserved_replica_then_completion(self) -> None:
        pool = GenerationPool([StubReplica(i) for i in range(4)])
        out = pool.generate("p", 8)
        assert out == [
            "r0_c0",
            "r0_c1",
            "r1_c0",
            "r1_c1",
            "r2_c0",
            "r2_c1",
            "r3_c0",
            "r3_c1",
        ]

    def test_last_terminated_aligns_with_completions(self) -> None:
        pool = GenerationPool([StubReplica(i) for i in range(2)])
        out = pool.generate("p", 6)  # 3 each
        assert len(pool.last_terminated) == len(out) == 6
        # each replica: [True, False, True]
        assert pool.last_terminated == [True, False, True, True, False, True]

    def test_uneven_split_skips_empty_replicas(self) -> None:
        replicas = [StubReplica(i) for i in range(4)]
        pool = GenerationPool(replicas)
        out = pool.generate("p", 2)  # only replicas 0,1 active
        assert out == ["r0_c0", "r1_c0"]

    def test_zero_n(self) -> None:
        pool = GenerationPool([StubReplica(0)])
        assert pool.generate("p", 0) == []
        assert pool.last_terminated == []

    def test_refresh_broadcasts_to_all_replicas(self) -> None:
        replicas = [StubReplica(i) for i in range(3)]
        pool = GenerationPool(replicas)

        class Trainer:
            def adapter_state_dict(self) -> dict[str, torch.Tensor]:
                return {"lora_A": torch.ones(2, 2)}

        pool.refresh_from(Trainer())
        for r in replicas:
            assert r.loaded_state is not None
            assert torch.equal(r.loaded_state["lora_A"], torch.ones(2, 2))

    def test_generate_runs_replicas_concurrently(self) -> None:
        active = {"n": 0, "peak": 0}
        lock = threading.Lock()

        class SlowReplica(StubReplica):
            def generate(self, prompt: str, n: int) -> list[str]:
                with lock:
                    active["n"] += 1
                    active["peak"] = max(active["peak"], active["n"])
                time.sleep(0.05)
                with lock:
                    active["n"] -= 1
                return super().generate(prompt, n)

        pool = GenerationPool([SlowReplica(i) for i in range(4)])
        pool.generate("p", 8)
        assert active["peak"] >= 2


def test_train_loop_uses_gen_pool(tmp_path: Any) -> None:
    from lethe.rl.train import GRPOTrainingLoop, TrainLoopConfig

    from .test_train_loop import BAD, GOOD, StubTrainablePolicy

    class PoolStub:
        def __init__(self) -> None:
            self.refreshed = 0
            self.gen_calls = 0
            self.last_terminated: list[bool] = []

        def refresh_from(self, trainer: Any) -> None:
            self.refreshed += 1

        def generate(self, prompt: str, n: int) -> list[str]:
            self.gen_calls += 1
            self.last_terminated = [True] * n
            return [GOOD if i % 2 == 0 else BAD for i in range(n)]

    policy = StubTrainablePolicy([GOOD, BAD])
    pool = PoolStub()

    def batch_scorer(sources: list[str]) -> list[dict[str, Any]]:
        return [
            {
                "status": "scored",
                "reward": 1.0 if "good" in s else 0.1,
                "compiled": True,
                "contracts_passed": "good" in s,
                "gates": {},
            }
            for s in sources
        ]

    config = TrainLoopConfig(
        n_per_prompt=4, total_steps=1, device="cpu", checkpoint_dir=str(tmp_path / "ckpt")
    )
    loop = GRPOTrainingLoop(config, policy, batch_scorer=batch_scorer, gen_pool=pool)
    metrics = loop.step()
    assert pool.refreshed == 1
    assert pool.gen_calls == 1
    assert metrics.mean_reward == pytest.approx((1.0 + 0.1 + 1.0 + 0.1) / 4)
