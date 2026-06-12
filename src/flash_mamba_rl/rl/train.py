"""End-to-end GRPO training loop over verifier-scored kernel generation.

One step: sample K completions for the op prompt → extract the final
fenced code block → score each source through the sandboxed op-harness
battery (``score_candidate_source``) → group-relative advantages →
clipped surrogate + KL loss → one optimizer step on the LoRA adapter.

Degenerate groups (all rewards identical — e.g. every candidate fails the
same way, or all saturate at the compile ceiling) carry no policy-gradient
signal; the step skips the log-prob forward passes and the update
entirely, recording ``loss=None``. The test is exact value equality, not a
float-std threshold: fp32 mean rounding gives an all-0.1 group a std of
~7e-9, which would otherwise turn every saturated group into a spurious
uniform-negative update.

Checkpointing is spot-box-shaped: step-stamped immutable adapter dirs via
peft ``save_pretrained``, with the atomically replaced ``trainer_state.pt``
as the commit point naming the valid adapter (step counter, optimizer
state, RNG states ride along). Resume = ``HFPolicy.from_pretrained(
adapter_path=GRPOTrainingLoop.latest_adapter_path(dir))`` +
:meth:`GRPOTrainingLoop.load_trainer_state`. Per-step metrics and
per-candidate rollout rows append to JSONL files in the checkpoint dir so
a detached box run can be polled with ``tail``; both use 1-based step ids.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import torch

from flash_mamba_rl.rl.grpo import compute_group_advantages, compute_grpo_loss
from flash_mamba_rl.rl.prompts import build_op_prompt

_CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

_OP_ENTRY_POINTS: dict[str, str] = {
    "forward_chunked_scan": "forward_chunked_scan",
    "elementwise_silu": "elementwise_silu",
    "backward_selective_scan": "backward_selective_scan",
    "mimo_backward": "mimo_backward",
    "complex_scan_rope": "complex_scan_rope",
    "fused_block_forward": "fused_block_forward",
    "fused_block_backward": "fused_block_backward",
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
        self,
        prompt: str,
        completions: list[str],
        *,
        use_adapter: bool = True,
        append_eos: bool | Sequence[bool] = True,
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
    score_fail_fast: bool = True
    reward_shaping: str = "none"
    measure_speedup: bool = False
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
        batch_scorer: Callable[[list[str]], list[dict[str, Any]]] | None = None,
    ) -> None:
        self.config = config
        self.policy = policy
        self.prompt = prompt if prompt is not None else build_op_prompt(config.op)
        self._scorer = scorer if scorer is not None else self._default_scorer
        self._batch_scorer = batch_scorer
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
            fail_fast=self.config.score_fail_fast,
            reward_shaping=self.config.reward_shaping,
            measure_speedup=self.config.measure_speedup,
        )

    def step(self) -> TrainStepMetrics:
        """One GRPO update. Dropout must be off — sampling and scoring
        log-probs must agree at step 0 (ratio identity)."""
        cfg = self.config
        self.policy.eval_mode()
        completions = self.policy.generate(self.prompt, cfg.n_per_prompt)

        sources = [extract_code(c, self._entry_point) for c in completions]
        to_score = [(idx, src) for idx, src in enumerate(sources) if src is not None]
        if self._batch_scorer is not None and to_score:
            scored = self._batch_scorer([src for _, src in to_score])
        else:
            scored = [self._scorer(src) for _, src in to_score]
        score_by_idx = {idx: score for (idx, _), score in zip(to_score, scored, strict=True)}

        rows: list[dict[str, Any]] = []
        for idx, source in enumerate(sources):
            if source is None:
                rows.append(
                    {
                        "step": self.step_idx + 1,
                        "idx": idx,
                        "status": "no_code_block",
                        "reward": 0.0,
                        "compiled": False,
                        "contracts_passed": False,
                    }
                )
            else:
                rows.append(
                    {
                        "step": self.step_idx + 1,
                        "idx": idx,
                        "source": source,
                        **score_by_idx[idx],
                    }
                )
        self._append_jsonl("rollouts.jsonl", rows)

        rewards = torch.tensor([float(r["reward"]) for r in rows], dtype=torch.float32)
        loss_val: float | None = None
        mean_kl: float | None = None
        grad_norm: float | None = None

        # Exact-equality test, not a float-std compare: eight identical 0.1
        # rewards carry std ~7e-9 from fp32 mean rounding, which would turn
        # every saturated group into a spurious uniform-negative update.
        if rewards.max().item() != rewards.min().item():
            advantages = compute_group_advantages(rewards)
            # EOS joins the scored trajectory only where the policy actually
            # stopped (HFPolicy.last_terminated); stubs without it score all.
            terminated = getattr(self.policy, "last_terminated", None)
            append_eos: bool | list[bool] = (
                list(terminated) if terminated and len(terminated) == len(completions) else True
            )
            with torch.no_grad():
                ref_lp, _ = self.policy.completion_log_probs(
                    self.prompt, completions, use_adapter=False, append_eos=append_eos
                )
            new_lp, mask = self.policy.completion_log_probs(
                self.prompt, completions, append_eos=append_eos
            )
            # One update per rollout = on-policy: the behaviour log-probs are
            # the current ones, detached. A multi-iteration update would need
            # a separate pre-update old pass.
            old_lp = new_lp.detach()
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
        """Adapter dirs are step-stamped and immutable; the atomically
        replaced ``trainer_state.pt`` is the commit point naming the valid
        one — a preemption mid-save leaves the previous checkpoint fully
        consistent (a half-written adapter dir is never referenced)."""
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
        """Restore step/optimizer/RNG if a checkpoint exists. Adapter weights
        are restored separately via ``HFPolicy.from_pretrained(adapter_path=
        latest_adapter_path(...))``."""
        path = os.path.join(self.config.checkpoint_dir, "trainer_state.pt")
        if not os.path.exists(path):
            return False
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.step_idx = int(state["step"])
        self.optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and "cuda_rng" in state:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        return True

    @staticmethod
    def latest_adapter_path(checkpoint_dir: str) -> str | None:
        """The adapter dir named by the committed trainer state, if any."""
        path = os.path.join(checkpoint_dir, "trainer_state.pt")
        if not os.path.exists(path):
            return None
        state = torch.load(path, map_location="cpu", weights_only=False)
        name = state.get("adapter_name")
        if not name:
            return None
        adapter = os.path.join(checkpoint_dir, str(name))
        return adapter if os.path.isdir(adapter) else None

    def _prune_adapters(self, *, keep: str) -> None:
        prefix = "adapter_step_"
        stamped = sorted(
            (d for d in os.listdir(self.config.checkpoint_dir) if d.startswith(prefix)),
            key=lambda d: int(d.removeprefix(prefix)),
        )
        # The committed one plus its predecessor (paranoia margin for spot).
        for name in stamped[:-2]:
            if name != keep:
                shutil.rmtree(os.path.join(self.config.checkpoint_dir, name), ignore_errors=True)

    def _append_jsonl(self, name: str, rows: list[dict[str, Any]]) -> None:
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        with open(os.path.join(self.config.checkpoint_dir, name), "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
