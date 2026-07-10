"""Warm-start SFT over verified targets."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import torch

from lethe.rl.prompts import build_op_prompt
from lethe.rl.sft_targets import available_targets, target_source, target_variants


class SFTPolicy(Protocol):
    """The policy surface SFT needs (HFPolicy satisfies it)."""

    def completion_log_probs(
        self,
        prompt: str,
        completions: list[str],
        *,
        use_adapter: bool = True,
        append_eos: bool | Sequence[bool] = True,
        temperature: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def trainable_parameters(self) -> Iterator[torch.nn.Parameter]: ...

    def save_adapter(self, path: str) -> None: ...

    def eval_mode(self) -> None: ...


@dataclass(frozen=True)
class SFTExample:
    op: str
    prompt: str
    completion: str
    variant: str = "eager"


def build_sft_examples(
    ops: Sequence[str] | None = None, variants: Sequence[str] | None = None
) -> list[SFTExample]:
    """One example per (op, available variant): op prompt → fenced verified target."""
    examples = []
    for op in ops if ops is not None else available_targets():
        present = target_variants(op)
        for variant in variants if variants is not None else present:
            if variant not in present:
                raise KeyError(f"{op} has no {variant!r} target")
            source = target_source(op, variant)
            if not source.endswith("\n"):
                source += "\n"
            examples.append(
                SFTExample(
                    op=op,
                    prompt=build_op_prompt(op),
                    completion=f"```python\n{source}```",
                    variant=variant,
                )
            )
    return examples


@dataclass(frozen=True)
class SFTConfig:
    total_steps: int = 300
    learning_rate: float = 1e-4
    max_grad_norm: float = 1.0
    checkpoint_dir: str = "checkpoints/sft"
    save_every: int = 10
    seed: int = 0
    # 1.0 makes the NLL exact cross-entropy, independent of the policy's sampling temperature.
    temperature: float = 1.0


@dataclass(frozen=True)
class SFTStepMetrics:
    """One row of the metrics JSONL."""

    step: int
    op: str
    variant: str
    loss: float
    tokens: int
    grad_norm: float


class SFTTrainingLoop:
    """Owns the optimizer and drives NLL step → checkpoint over the examples."""

    def __init__(self, config: SFTConfig, policy: SFTPolicy, examples: list[SFTExample]) -> None:
        if not examples:
            raise ValueError("examples must be non-empty")
        if config.temperature <= 0.0:
            raise ValueError(f"SFT temperature must be positive, got {config.temperature}")
        self.config = config
        self.policy = policy
        self.examples = examples
        self.optimizer = torch.optim.AdamW(
            list(policy.trainable_parameters()), lr=config.learning_rate
        )
        self.step_idx = 0

    def _example_for_step(self, step_idx: int) -> SFTExample:
        """Seeded per-epoch shuffle, a pure function of (seed, epoch, position)."""
        n = len(self.examples)
        epoch, pos = divmod(step_idx, n)
        gen = torch.Generator()
        gen.manual_seed(self.config.seed * 1_000_003 + epoch)
        order = torch.randperm(n, generator=gen)
        return self.examples[int(order[pos].item())]

    def step(self) -> SFTStepMetrics:
        cfg = self.config
        self.policy.eval_mode()
        example = self._example_for_step(self.step_idx)
        # The target ends at the closing fence by choice; append_eos trains the stop decision too.
        log_probs, mask = self.policy.completion_log_probs(
            example.prompt, [example.completion], append_eos=True, temperature=cfg.temperature
        )
        tokens = int(mask.sum().item())
        loss = -(log_probs * mask).sum() / mask.sum().clamp(min=1)
        self.optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        total_norm = torch.nn.utils.clip_grad_norm_(
            list(self.policy.trainable_parameters()), cfg.max_grad_norm
        )
        self.optimizer.step()

        self.step_idx += 1
        metrics = SFTStepMetrics(
            step=self.step_idx,
            op=example.op,
            variant=example.variant,
            loss=float(loss.item()),
            tokens=tokens,
            grad_norm=float(total_norm.item()),
        )
        self._append_jsonl("metrics.jsonl", [asdict(metrics)])
        return metrics

    def run(self) -> list[SFTStepMetrics]:
        history: list[SFTStepMetrics] = []
        while self.step_idx < self.config.total_steps:
            metrics = self.step()
            history.append(metrics)
            print(
                f"step {metrics.step}/{self.config.total_steps} "
                f"op={metrics.op}[{metrics.variant}] "
                f"loss={metrics.loss:.4f} tokens={metrics.tokens} "
                f"grad_norm={metrics.grad_norm:.3f}",
                flush=True,
            )
            if self.step_idx % self.config.save_every == 0:
                self.save_checkpoint()
        if self.step_idx % self.config.save_every != 0:
            self.save_checkpoint()
        return history

    def save_checkpoint(self) -> None:
        cfg = self.config
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        adapter_name = f"adapter_step_{self.step_idx}"
        self.policy.save_adapter(os.path.join(cfg.checkpoint_dir, adapter_name))
        state: dict[str, Any] = {
            "step": self.step_idx,
            "adapter_name": adapter_name,
            "optimizer": self.optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda_rng"] = torch.cuda.get_rng_state_all()
        tmp = os.path.join(cfg.checkpoint_dir, "trainer_state.pt.tmp")
        torch.save(state, tmp)
        os.replace(tmp, os.path.join(cfg.checkpoint_dir, "trainer_state.pt"))
        self._prune_adapters(keep=adapter_name)

    def load_trainer_state(self) -> bool:
        path = os.path.join(self.config.checkpoint_dir, "trainer_state.pt")
        if not os.path.exists(path):
            return False
        # weights_only=True closes the pickle-RCE on resume from a foreign checkpoint dir.
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.step_idx = int(state["step"])
        self.optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and "cuda_rng" in state:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        return True

    def _prune_adapters(self, *, keep: str) -> None:
        prefix = "adapter_step_"
        stamped = sorted(
            (d for d in os.listdir(self.config.checkpoint_dir) if d.startswith(prefix)),
            key=lambda d: int(d.removeprefix(prefix)),
        )
        for name in stamped[:-2]:
            if name != keep:
                shutil.rmtree(os.path.join(self.config.checkpoint_dir, name), ignore_errors=True)

    def _append_jsonl(self, name: str, rows: list[dict[str, Any]]) -> None:
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        with open(os.path.join(self.config.checkpoint_dir, name), "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
