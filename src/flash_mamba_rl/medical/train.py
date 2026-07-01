"""PTB-XL multi-label training loop (Phase F.3).

Framework-free: a plain AdamW + ``BCEWithLogitsLoss`` loop over a
``Mamba3ECGClassifier``, with macro-AUC evaluation and spot-resilient
checkpoint/resume. DDP-aware but DDP-optional — single-process on CPU (the
test path), data-parallel across the 8 B200 when launched under
``torch.distributed`` (the box path, ``scratch/ptbxl_train.py``).

Checkpointing mirrors ``rl/train.py``: a step-stamped immutable model file
plus the atomically replaced ``trainer_state.pt`` as the commit point that
names the valid model (step, optimizer, RNG ride along). A preemption
mid-save leaves the previous checkpoint fully consistent. Only rank 0
writes; all ranks barrier so resume is collective.

macro-AUC is owned here (rank-based Mann-Whitney, ties averaged) rather than
pulled from sklearn: it must skip degenerate classes (all-positive or
all-negative in the eval split) which ``roc_auc_score`` raises on, and the
loop stays import-light.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["MedicalTrainConfig", "MedicalTrainer", "macro_auc"]


def _is_dist() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _average_ranks(x: Tensor) -> Tensor:
    """1-based ranks of *x* with tied values assigned their group's mean rank."""
    _uniq, inverse, counts = torch.unique(x, return_inverse=True, return_counts=True)
    cum = torch.cumsum(counts, 0)
    start = (cum - counts).to(torch.float64)
    avg_rank = start + (counts.to(torch.float64) + 1.0) / 2.0
    return cast(Tensor, avg_rank[inverse])


def _binary_auc(scores: Tensor, labels: Tensor) -> float | None:
    """ROC-AUC for one class via the rank statistic; None if degenerate."""
    n_pos = float(labels.sum().item())
    n_neg = float(labels.numel()) - n_pos
    if n_pos == 0.0 or n_neg == 0.0:
        return None
    if not torch.isfinite(scores).all():
        return None  # NaN/Inf logits sort unpredictably -> skip rather than emit AUC ∉ [0,1]
    ranks = _average_ranks(scores.to(torch.float64))
    pos_rank_sum = float(ranks[labels > 0.5].sum().item())
    return (pos_rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)


def macro_auc(logits: Tensor, labels: Tensor) -> float:
    """Mean per-class ROC-AUC over classes with both labels present.

    logits, labels: ``[N, C]``. Returns ``nan`` if no class is evaluable.
    """
    per_class = [
        auc
        for c in range(logits.shape[1])
        if (auc := _binary_auc(logits[:, c], labels[:, c])) is not None
    ]
    if not per_class:
        return float("nan")
    return sum(per_class) / len(per_class)


@dataclass(frozen=True)
class MedicalTrainConfig:
    """Hyperparameters + plumbing for :class:`MedicalTrainer`."""

    total_steps: int = 1000
    learning_rate: float = 3e-4
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    # #19: cosine LR decay after warmup, peak -> min_lr_ratio*peak over the
    # remaining steps. Default off keeps the flat-after-warmup schedule the
    # existing tests pin.
    lr_decay: bool = False
    min_lr_ratio: float = 0.1
    device: str = "cpu"
    checkpoint_dir: str = "checkpoints/medical"
    save_every: int = 100
    eval_every: int = 100
    log_every: int = 10


class MedicalTrainer:
    """Owns the model, optimizer and loss; drives step → eval → checkpoint.

    Under ``torch.distributed`` the model is wrapped in ``DistributedDataParallel``
    and only rank 0 writes checkpoints/metrics; evaluation all-gathers per-rank
    logits so the macro-AUC is over the full eval split. Off distributed it is a
    plain single-process loop.
    """

    def __init__(self, model: nn.Module, config: MedicalTrainConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self._dist = _is_dist()
        self.rank = torch.distributed.get_rank() if self._dist else 0
        self.world_size = torch.distributed.get_world_size() if self._dist else 1

        model = model.to(self.device)
        self._core = model
        if self._dist:
            ddp_kwargs: dict[str, Any] = {}
            if self.device.type == "cuda":
                ddp_kwargs["device_ids"] = [self.device.index]
            self.model: nn.Module = nn.parallel.DistributedDataParallel(model, **ddp_kwargs)
        else:
            self.model = model

        self.optimizer = torch.optim.AdamW(
            self._core.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.criterion = nn.BCEWithLogitsLoss()
        # The model may be cast to bf16 (the box path); the fp32 ECG inputs must
        # match its dtype or the first Linear's matmul dtype-mismatches. Loss is
        # computed in fp32 regardless (logits upcast) for stable BCE.
        self._in_dtype = next(self._core.parameters()).dtype
        self.step_idx = 0

    def train_step(self, ecg: Tensor, labels: Tensor) -> tuple[float, float]:
        """One optimizer step. Returns ``(loss, grad_norm)``."""
        self.model.train()
        # Stateless schedule (derived from the checkpointed step_idx): linear
        # warmup — a 1.1B SSM from scratch at full LR spikes the first updates
        # into NaN — then optional cosine decay to min_lr_ratio*peak (#19).
        cfg = self.config
        warm = cfg.warmup_steps
        step = self.step_idx + 1
        if warm and step <= warm:
            lr = cfg.learning_rate * step / warm
        elif cfg.lr_decay and cfg.total_steps > warm:
            progress = min(1.0, max(0.0, (step - warm) / max(1, cfg.total_steps - warm)))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = cfg.learning_rate * (cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine)
        else:
            lr = cfg.learning_rate
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        ecg = ecg.to(self.device, self._in_dtype)
        labels = labels.to(self.device)
        self.optimizer.zero_grad()
        logits = self.model(ecg)
        loss = self.criterion(logits.float(), labels.float())
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self._core.parameters(), self.config.max_grad_norm
        )
        # Skip the update on a non-finite grad: one bad batch (forward overflow on
        # an outlier ECG) otherwise applies a NaN step that poisons every weight,
        # and the loss is NaN forever after. Skipping keeps the model in the
        # last-good state; the next batch recovers.
        if torch.isfinite(grad_norm):
            self.optimizer.step()
        self.step_idx += 1
        return float(loss.item()), float(grad_norm.item())

    @torch.no_grad()
    def evaluate(self, loader: Iterable[tuple[Tensor, Tensor]]) -> dict[str, float]:
        """Mean BCE loss + macro-AUC over *loader* (all-gathered across ranks).

        Under DDP the caller MUST pass a non-overlapping shard per rank (e.g. a
        ``DistributedSampler`` with ``drop_last`` or a manual slice). This gathers
        and concatenates each rank's logits with no dedupe, so a default padded
        sampler double-counts the eval tail and biases the metrics.
        """
        self.model.eval()
        logits_chunks: list[Tensor] = []
        label_chunks: list[Tensor] = []
        loss_sum = 0.0
        n = 0
        for ecg, labels in loader:
            ecg = ecg.to(self.device, self._in_dtype)
            labels = labels.to(self.device)
            logits = self.model(ecg)
            bs = labels.shape[0]
            loss_sum += float(self.criterion(logits.float(), labels.float()).item()) * bs
            n += bs
            logits_chunks.append(logits.detach().cpu())
            label_chunks.append(labels.detach().cpu())

        logits = torch.cat(logits_chunks, dim=0)
        labels = torch.cat(label_chunks, dim=0)
        if self._dist:
            logits = torch.cat(self._all_gather(logits), dim=0)
            labels = torch.cat(self._all_gather(labels), dim=0)
            # all_reduce must run on the NCCL device — a CPU tensor raises
            # "No backend type associated with device type cpu" under nccl.
            totals = torch.tensor([loss_sum, float(n)], dtype=torch.float64, device=self.device)
            torch.distributed.all_reduce(totals)
            loss_sum, n = float(totals[0].item()), int(totals[1].item())
        return {"loss": loss_sum / max(n, 1), "macro_auc": macro_auc(logits, labels)}

    @staticmethod
    def _all_gather(t: Tensor) -> list[Tensor]:
        gathered: list[Any] = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, t)
        return [g for g in gathered if isinstance(g, Tensor)]

    def run(
        self,
        train_loader: Iterable[tuple[Tensor, Tensor]],
        val_loader: Iterable[tuple[Tensor, Tensor]] | None = None,
    ) -> None:
        """Train to ``total_steps``, cycling *train_loader*, eval/ckpt on schedule."""
        cfg = self.config
        data = _cycle(train_loader)
        while self.step_idx < cfg.total_steps:
            ecg, labels = next(data)
            loss, grad_norm = self.train_step(ecg, labels)
            if self.rank == 0 and self.step_idx % cfg.log_every == 0:
                self._log({"step": self.step_idx, "loss": loss, "grad_norm": grad_norm})
            if val_loader is not None and self.step_idx % cfg.eval_every == 0:
                metrics = self.evaluate(val_loader)
                if self.rank == 0:
                    self._log({"step": self.step_idx, **metrics, "split": "val"})
            if self.step_idx % cfg.save_every == 0:
                self.save_checkpoint()

    # ------------------------------------------------------------------
    # Checkpointing (spot box: model + optimizer + step + RNG)
    # ------------------------------------------------------------------

    def _local_rng_state(self) -> dict[str, Tensor]:
        """This rank's CPU RNG (+ its *own* CUDA device's RNG, if on cuda)."""
        rng: dict[str, Tensor] = {"cpu": torch.get_rng_state()}
        if self.device.type == "cuda":
            rng["cuda"] = torch.cuda.get_rng_state(self.device)
        return rng

    def _all_rng_states(self) -> list[dict[str, Tensor]]:
        """Per-rank RNG states, index = rank (collective; every rank participates)."""
        local = self._local_rng_state()
        if not self._dist:
            return [local]
        gathered: list[dict[str, Tensor]] = [{} for _ in range(self.world_size)]
        torch.distributed.all_gather_object(gathered, local)
        return gathered

    def save_checkpoint(self) -> None:
        """Step-stamped model file + atomic ``trainer_state.pt`` commit point."""
        cfg = self.config
        # Per-rank RNG is gathered on every rank (a collective), then written by
        # rank 0 — restoring the *current device's* stream per rank, so resume
        # neither depends on the machine's GPU count nor collapses all ranks onto
        # rank 0's RNG (identical dropout masks).
        rng_states = self._all_rng_states()
        if self.rank == 0:
            os.makedirs(cfg.checkpoint_dir, exist_ok=True)
            model_name = f"model_step_{self.step_idx}.pt"
            torch.save(self._core.state_dict(), os.path.join(cfg.checkpoint_dir, model_name))
            state: dict[str, Any] = {
                "step": self.step_idx,
                "model_name": model_name,
                "optimizer": self.optimizer.state_dict(),
                "rng_states": rng_states,
            }
            tmp = os.path.join(cfg.checkpoint_dir, "trainer_state.pt.tmp")
            torch.save(state, tmp)
            os.replace(tmp, os.path.join(cfg.checkpoint_dir, "trainer_state.pt"))
            self._prune_models(keep=model_name)
        if self._dist:
            torch.distributed.barrier()

    def load_checkpoint(self) -> bool:
        """Restore model/step/optimizer/RNG from the committed checkpoint."""
        path = os.path.join(self.config.checkpoint_dir, "trainer_state.pt")
        if not os.path.exists(path):
            return False
        # weights_only=True: the checkpoint holds only step/model_name plus the
        # optimizer state_dict and per-rank RNG tensors — all tensors + primitive
        # containers, which the safe loader accepts. It closes the pickle-RCE if
        # checkpoint_dir ever points at a run dir this process did not write.
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.step_idx = int(state["step"])
        model_path = os.path.join(self.config.checkpoint_dir, str(state["model_name"]))
        self._core.load_state_dict(torch.load(model_path, map_location=self.device))
        self.optimizer.load_state_dict(state["optimizer"])
        self._restore_rng_state(state["rng_states"])
        return True

    def _restore_rng_state(self, rng_states: list[dict[str, Tensor]]) -> None:
        if self.rank >= len(rng_states):
            return  # resumed with more ranks than were saved — keep the seeded RNG
        local = rng_states[self.rank]
        torch.set_rng_state(local["cpu"])
        if self.device.type == "cuda" and "cuda" in local:
            torch.cuda.set_rng_state(local["cuda"], self.device)

    def _prune_models(self, *, keep: str) -> None:
        prefix = "model_step_"
        stamped = sorted(
            (d for d in os.listdir(self.config.checkpoint_dir) if d.startswith(prefix)),
            key=lambda d: int(d.removeprefix(prefix).removesuffix(".pt")),
        )
        for name in stamped[:-2]:
            if name != keep:
                path = os.path.join(self.config.checkpoint_dir, name)
                try:
                    os.remove(path)
                except OSError:
                    shutil.rmtree(path, ignore_errors=True)

    def _log(self, row: dict[str, Any]) -> None:
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        with open(
            os.path.join(self.config.checkpoint_dir, "metrics.jsonl"), "a", encoding="utf-8"
        ) as f:
            f.write(json.dumps(row) + "\n")
        print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)


def _cycle(loader: Iterable[tuple[Tensor, Tensor]]) -> Iterator[tuple[Tensor, Tensor]]:
    while True:
        empty = True
        for batch in loader:
            empty = False
            yield batch
        if empty:
            raise ValueError("train_loader yielded no batches")
