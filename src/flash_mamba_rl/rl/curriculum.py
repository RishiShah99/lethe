"""Six-level curriculum over the Phase C op suite with promotion gates.

Level order is forward ops first (single gate view, densest reward
signal), then backward ops by view count — the dependency structure of
the kernels themselves, not their Phase C build order.

Promotion: mean group reward >= ``promote_threshold`` for
``promote_window`` consecutive steps. A level that exhausts
``max_steps_per_level`` without promotion advances anyway with
``promoted=False`` recorded: the kill criterion is evaluated per-op over
the whole run, and a stuck level must not starve later levels of
training signal — the honest negative stays in the level record.

``CurriculumSchedule`` is pure bookkeeping (no torch); ``CurriculumRunner``
drives one ``GRPOTrainingLoop`` per level over a shared policy — LoRA
weights carry across levels, the optimizer restarts fresh per level
(stale Adam moments from one task are noise for the next). Schedule
state is written atomically to ``curriculum_state.json`` after every
step so a spot preemption resumes mid-level: the schedule names the
level, the level's own trainer_state.pt names the step and adapter.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any

from flash_mamba_rl.rl.train import GRPOTrainingLoop, TrainablePolicy, TrainLoopConfig

DEFAULT_CURRICULUM: tuple[str, ...] = (
    "forward_chunked_scan",
    "complex_scan_rope",
    "fused_block_forward",
    "backward_selective_scan",
    "mimo_backward",
    "fused_block_backward",
)


@dataclass(frozen=True)
class CurriculumConfig:
    ops: tuple[str, ...] = DEFAULT_CURRICULUM
    promote_threshold: float = 0.35
    promote_window: int = 8
    max_steps_per_level: int = 200


@dataclass
class LevelState:
    """One curriculum level's running record."""

    op: str
    steps: int = 0
    consecutive_at_threshold: int = 0
    promoted: bool = False
    closed: bool = False
    best_mean_reward: float = 0.0
    best_max_reward: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class CurriculumSchedule:
    """Promotion-gate bookkeeping over an ordered op list."""

    def __init__(self, config: CurriculumConfig | None = None) -> None:
        self.config = config if config is not None else CurriculumConfig()
        self.levels = [LevelState(op=op) for op in self.config.ops]
        self.level_idx = 0

    @property
    def done(self) -> bool:
        return self.level_idx >= len(self.levels)

    @property
    def current_level(self) -> LevelState:
        return self.levels[self.level_idx]

    @property
    def current_op(self) -> str:
        return self.current_level.op

    def record_step(self, mean_reward: float, max_reward: float = 0.0) -> bool:
        """Record one training step's group stats; True when the level closed."""
        cfg = self.config
        level = self.current_level
        level.steps += 1
        level.best_mean_reward = max(level.best_mean_reward, mean_reward)
        level.best_max_reward = max(level.best_max_reward, max_reward)
        if mean_reward >= cfg.promote_threshold:
            level.consecutive_at_threshold += 1
        else:
            level.consecutive_at_threshold = 0
        if level.consecutive_at_threshold >= cfg.promote_window:
            level.promoted = True
            level.closed = True
        elif level.steps >= cfg.max_steps_per_level:
            level.closed = True
        if level.closed:
            self.level_idx += 1
        return level.closed

    def state_dict(self) -> dict[str, Any]:
        return {
            "level_idx": self.level_idx,
            "levels": [lv.as_dict() for lv in self.levels],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.level_idx = int(state["level_idx"])
        saved = {lv["op"]: lv for lv in state["levels"]}
        for level in self.levels:
            if level.op in saved:
                for key, value in saved[level.op].items():
                    if key != "op":
                        setattr(level, key, value)

    def summary(self) -> list[dict[str, Any]]:
        return [lv.as_dict() for lv in self.levels]


@dataclass
class CurriculumRunner:
    """Drives one GRPO loop per curriculum level over a shared policy.

    ``scorer_factory`` (op name -> scorer callable) replaces the default
    sandboxed scorer in tests and in multi-GPU runs where scoring is
    farmed out per device.
    """

    base_config: TrainLoopConfig
    policy: TrainablePolicy
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    scorer_factory: Any = None

    def __post_init__(self) -> None:
        self.schedule = CurriculumSchedule(self.curriculum)

    @property
    def state_path(self) -> str:
        return os.path.join(self.base_config.checkpoint_dir, "curriculum_state.json")

    def level_dir(self, idx: int) -> str:
        return os.path.join(
            self.base_config.checkpoint_dir, f"level{idx}_{self.curriculum.ops[idx]}"
        )

    def level_config(self, idx: int) -> TrainLoopConfig:
        return dataclasses.replace(
            self.base_config,
            op=self.curriculum.ops[idx],
            checkpoint_dir=self.level_dir(idx),
            total_steps=self.curriculum.max_steps_per_level,
        )

    def resume(self) -> bool:
        """Restore schedule state if a prior run left one."""
        if not os.path.exists(self.state_path):
            return False
        with open(self.state_path, encoding="utf-8") as f:
            self.schedule.load_state_dict(json.load(f))
        return True

    def _write_state(self) -> None:
        os.makedirs(self.base_config.checkpoint_dir, exist_ok=True)
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.schedule.state_dict(), f, indent=2)
        os.replace(tmp, self.state_path)

    def _make_loop(self, idx: int) -> GRPOTrainingLoop:
        config = self.level_config(idx)
        scorer = self.scorer_factory(config.op) if self.scorer_factory is not None else None
        loop = GRPOTrainingLoop(config, self.policy, scorer=scorer)
        loop.load_trainer_state()
        return loop

    def run(self) -> list[dict[str, Any]]:
        """Train through the curriculum; returns the per-level summary."""
        while not self.schedule.done:
            idx = self.schedule.level_idx
            loop = self._make_loop(idx)
            level = self.schedule.current_level
            # A resumed level replays its already-recorded steps inside
            # trainer_state; the schedule only counts new ones.
            while not level.closed and loop.step_idx < loop.config.total_steps:
                metrics = loop.step()
                closed = self.schedule.record_step(metrics.mean_reward, metrics.max_reward)
                loop.save_checkpoint()
                self._write_state()
                print(
                    f"[curriculum] level {idx} ({level.op}) step {level.steps} "
                    f"mean_r={metrics.mean_reward:.3f} "
                    f"consec={level.consecutive_at_threshold} "
                    f"promoted={level.promoted}",
                    flush=True,
                )
                if closed:
                    break
            if not level.closed:
                # Trainer hit total_steps without the schedule closing the
                # level (resume drift) — close it as unpromoted.
                level.closed = True
                self.schedule.level_idx += 1
                self._write_state()
        return self.schedule.summary()
