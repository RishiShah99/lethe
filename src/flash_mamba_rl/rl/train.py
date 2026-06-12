"""End-to-end GRPO training loop over verifier-scored kernel generation.

One step: sample K completions for the op prompt → extract the final
fenced code block → score each source through the sandboxed op-harness
battery (``score_candidate_source``) → group-relative advantages →
clipped surrogate + KL loss → one optimizer step on the LoRA adapter.

Degenerate groups (zero reward variance — e.g. every candidate fails
identically) carry no policy-gradient signal; the step skips all three
log-prob forward passes and the update entirely, recording ``loss=None``.

Checkpointing is spot-box-shaped: adapter weights via peft
``save_pretrained`` plus a ``trainer_state.pt`` (step counter, optimizer
state, RNG states). Resume = ``HFPolicy.from_pretrained(adapter_path=...)``
+ :meth:`GRPOTrainingLoop.load_trainer_state`. Per-step metrics and
per-candidate rollout rows append to JSONL files in the checkpoint dir so
a detached box run can be polled with ``tail``.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import torch

from flash_mamba_rl.rl.grpo import compute_group_advantages, compute_grpo_loss
from flash_mamba_rl.rl.prompts import build_op_prompt

_CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

_OP_ENTRY_POINTS: dict[str, str] = {
    "forward_chunked_scan": "forward_chunked_scan",
    "elementwise_silu": "elementwise_silu",
}


def extract_code(completion: str, entry_point: str) -> str | None:
    """Final fenced code block defining *entry_point*; falls back to the last block."""
    blocks = _CODE_BLOCK.findall(completion)
    for block in reversed(blocks):
        if f"def {entry_point}" in block:
            return str(block)
    return str(blocks[-1]) if blocks else None


class TrainablePolicy(Protocol):
    """The policy surface the training loop needs (HFPolicy satisfies it)."""

    def generate(self, prompt: str, n: int) -> list[str]: ...

    def completion_log_probs(
        self, prompt: str, completions: list[str], *, use_adapter: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def trainable_parameters(self) -> Iterator[torch.nn.Parameter]: ...

    def save_adapter(self, path: str) -> None: ...

    def eval_mode(self) -> None: ...


@dataclass(frozen=True)
class TrainLoopConfig:
    """Hyperparameters + plumbing for :class:`GRPOTrainingLoop`."""

    op: str = "forward_chunked_scan"
    n_per_prompt: int = 8
    total_steps: int = 100
    learning_rate: float = 1e-5
    clip_eps: float = 0.2
    kl_coef: float = 0.04
    max_grad_norm: float = 1.0
    device: str = "cuda"
    score_timeout_s: float = 420.0
    checkpoint_dir: str = "checkpoints/grpo"
    save_every: int = 1


@dataclass(frozen=True)
class TrainStepMetrics:
    """One row of the metrics JSONL."""

    step: int
    mean_reward: float
    max_reward: float
    n_no_code: int
    n_compiled: int
    n_contracts_passed: int
    loss: float | None
    mean_kl: float | None
    grad_norm: float | None


class GRPOTrainingLoop:
    """Owns the optimizer and drives generate → score → update → checkpoint."""

    def __init__(
        self,
        config: TrainLoopConfig,
        policy: TrainablePolicy,
        *,
        prompt: str | None = None,
        scorer: Callable[[str], dict[str, Any]] | None = None,
    ) -> None:
        self.config = config
        self.policy = policy
        self.prompt = prompt if prompt is not None else build_op_prompt(config.op)
        self._scorer = scorer if scorer is not None else self._default_scorer
        self._entry_point = _OP_ENTRY_POINTS[config.op]
        self.optimizer = torch.optim.AdamW(
            list(policy.trainable_parameters()), lr=config.learning_rate
        )
        self.step_idx = 0

    def _default_scorer(self, source: str) -> dict[str, Any]:
        from flash_mamba_rl.verifier.candidate_scoring import score_candidate_source

        return score_candidate_source(
            source,
            op=self.config.op,
            device=self.config.device,
            timeout_s=self.config.score_timeout_s,
        )

    def step(self) -> TrainStepMetrics:
        """One GRPO update. Dropout must be off — sampling and scoring
        log-probs must agree at step 0 (ratio identity)."""
        cfg = self.config
        self.policy.eval_mode()
        completions = self.policy.generate(self.prompt, cfg.n_per_prompt)

        rows: list[dict[str, Any]] = []
        for idx, completion in enumerate(completions):
            source = extract_code(completion, self._entry_point)
            if source is None:
                rows.append(
                    {
                        "step": self.step_idx,
                        "idx": idx,
                        "status": "no_code_block",
                        "reward": 0.0,
                        "compiled": False,
                        "contracts_passed": False,
                    }
                )
            else:
                score = self._scorer(source)
                rows.append(
                    {
                        "step": self.step_idx,
                        "idx": idx,
                        "source": source,
                        **score,
                    }
                )
        self._append_jsonl("rollouts.jsonl", rows)

        rewards = torch.tensor([float(r["reward"]) for r in rows], dtype=torch.float32)
        loss_val: float | None = None
        mean_kl: float | None = None
        grad_norm: float | None = None

        if rewards.std(correction=0).item() > 0.0:
            advantages = compute_group_advantages(rewards)
            with torch.no_grad():
                old_lp, mask = self.policy.completion_log_probs(self.prompt, completions)
                ref_lp, _ = self.policy.completion_log_probs(
                    self.prompt, completions, use_adapter=False
                )
            new_lp, _ = self.policy.completion_log_probs(self.prompt, completions)
            device = new_lp.device
            loss = compute_grpo_loss(
                advantages.to(device),
                new_lp,
                old_lp,
                ref_lp,
                clip_eps=cfg.clip_eps,
                kl_coef=cfg.kl_coef,
                mask=mask,
            )
            self.optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            total_norm = torch.nn.utils.clip_grad_norm_(
                list(self.policy.trainable_parameters()), cfg.max_grad_norm
            )
            self.optimizer.step()

            with torch.no_grad():
                d = ref_lp - new_lp
                kl = (torch.exp(d) - d - 1.0) * mask
                mean_kl = float(kl.sum().item() / mask.sum().clamp(min=1).item())
            loss_val = float(loss.item())
            grad_norm = float(total_norm.item())

        self.step_idx += 1
        metrics = TrainStepMetrics(
            step=self.step_idx,
            mean_reward=float(rewards.mean().item()),
            max_reward=float(rewards.max().item()),
            n_no_code=sum(1 for r in rows if r["status"] == "no_code_block"),
            n_compiled=sum(1 for r in rows if r.get("compiled")),
            n_contracts_passed=sum(1 for r in rows if r.get("contracts_passed")),
            loss=loss_val,
            mean_kl=mean_kl,
            grad_norm=grad_norm,
        )
        self._append_jsonl("metrics.jsonl", [asdict(metrics)])
        return metrics

    def run(self) -> list[TrainStepMetrics]:
        """Train to ``total_steps``, checkpointing every ``save_every`` steps."""
        history: list[TrainStepMetrics] = []
        while self.step_idx < self.config.total_steps:
            metrics = self.step()
            history.append(metrics)
            print(
                f"step {metrics.step}/{self.config.total_steps} "
                f"mean_r={metrics.mean_reward:.3f} max_r={metrics.max_reward:.3f} "
                f"contracts={metrics.n_contracts_passed} loss={metrics.loss} "
                f"kl={metrics.mean_kl}",
                flush=True,
            )
            if self.step_idx % self.config.save_every == 0:
                self.save_checkpoint()
        return history

    # ------------------------------------------------------------------
    # Checkpointing (spot box: adapter + optimizer + step + RNG)
    # ------------------------------------------------------------------

    def save_checkpoint(self) -> None:
        cfg = self.config
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        self.policy.save_adapter(os.path.join(cfg.checkpoint_dir, "adapter"))
        state: dict[str, Any] = {
            "step": self.step_idx,
            "optimizer": self.optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda_rng"] = torch.cuda.get_rng_state_all()
        tmp = os.path.join(cfg.checkpoint_dir, "trainer_state.pt.tmp")
        torch.save(state, tmp)
        os.replace(tmp, os.path.join(cfg.checkpoint_dir, "trainer_state.pt"))

    def load_trainer_state(self) -> bool:
        """Restore step/optimizer/RNG if a checkpoint exists. Adapter weights
        are restored separately via ``HFPolicy.from_pretrained(adapter_path=...)``."""
        path = os.path.join(self.config.checkpoint_dir, "trainer_state.pt")
        if not os.path.exists(path):
            return False
        state = torch.load(path, weights_only=False)
        self.step_idx = int(state["step"])
        self.optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and "cuda_rng" in state:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        return True

    def _append_jsonl(self, name: str, rows: list[dict[str, Any]]) -> None:
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        with open(os.path.join(self.config.checkpoint_dir, name), "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
