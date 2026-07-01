"""CPU smoke for the PTB-XL trainer (Phase F.3).

No real PTB-XL files: a synthetic in-memory dataset whose labels are a linear
readout of the signal, so the tiny model can actually fit it. Pins one-step loss
descent, sane macro-AUC, and checkpoint→resume of step+optimizer+model.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from flash_mamba_rl.medical.model import Mamba3Config, Mamba3ECGClassifier
from flash_mamba_rl.medical.train import (
    MedicalTrainConfig,
    MedicalTrainer,
    macro_auc,
)

_T = 64  # divisible by tiny()'s chunk_size=8
_N_CLASSES = 5


class _SyntheticECG(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Fittable: labels are a fixed-projection threshold of the signal mean."""

    def __init__(self, n: int, *, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.signals = torch.randn(n, 12, _T, generator=g)
        proj = torch.randn(12, _N_CLASSES, generator=g)
        feats = self.signals.mean(dim=2) @ proj  # [n, C]
        self.labels = (feats > feats.median(dim=0, keepdim=True).values).float()

    def __len__(self) -> int:
        return self.signals.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.signals[idx], self.labels[idx]


def _trainer(tmp_path: object, **overrides: object) -> MedicalTrainer:
    torch.manual_seed(0)
    model = Mamba3ECGClassifier(Mamba3Config.tiny())
    cfg = MedicalTrainConfig(
        device="cpu",
        checkpoint_dir=str(tmp_path),
        total_steps=int(overrides.get("total_steps", 5)),  # type: ignore[arg-type]
        learning_rate=float(overrides.get("learning_rate", 1e-2)),  # type: ignore[arg-type]
    )
    return MedicalTrainer(model, cfg)


class TestLRSchedule:
    def test_cosine_decay_after_warmup(self, tmp_path: object) -> None:
        torch.manual_seed(0)
        model = Mamba3ECGClassifier(Mamba3Config.tiny())
        cfg = MedicalTrainConfig(
            device="cpu",
            checkpoint_dir=str(tmp_path),
            total_steps=12,
            warmup_steps=3,
            learning_rate=1.0,
            lr_decay=True,
            min_lr_ratio=0.1,
        )
        trainer = MedicalTrainer(model, cfg)
        ecg = torch.randn(2, 12, 64)
        labels = (torch.rand(2, 5) > 0.5).float()
        lrs: list[float] = []
        for _ in range(12):
            trainer.train_step(ecg, labels)
            lrs.append(trainer.optimizer.param_groups[0]["lr"])
        assert lrs[0] < lrs[2]  # warmup ramps up
        assert abs(lrs[2] - 1.0) < 1e-9  # peak at end of warmup
        post = lrs[3:]
        assert all(post[i] >= post[i + 1] - 1e-9 for i in range(len(post) - 1))  # monotone decay
        assert lrs[-1] < 0.5 and lrs[-1] >= 0.1 - 1e-6  # decays toward min_lr_ratio*peak

    def test_decay_off_is_flat_after_warmup(self, tmp_path: object) -> None:
        torch.manual_seed(0)
        model = Mamba3ECGClassifier(Mamba3Config.tiny())
        cfg = MedicalTrainConfig(
            device="cpu",
            checkpoint_dir=str(tmp_path),
            total_steps=12,
            warmup_steps=2,
            learning_rate=1.0,
        )
        trainer = MedicalTrainer(model, cfg)
        ecg = torch.randn(2, 12, 64)
        labels = (torch.rand(2, 5) > 0.5).float()
        lrs: list[float] = []
        for _ in range(6):
            trainer.train_step(ecg, labels)
            lrs.append(trainer.optimizer.param_groups[0]["lr"])
        assert all(abs(x - 1.0) < 1e-9 for x in lrs[2:])  # flat at peak (default off)


class TestMacroAUC:
    def test_perfect_ranking_is_one(self) -> None:
        labels = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
        logits = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]])
        assert macro_auc(logits, labels) == 1.0

    def test_inverted_ranking_is_zero(self) -> None:
        labels = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
        logits = torch.tensor([[2.0], [1.0], [-1.0], [-2.0]])
        assert macro_auc(logits, labels) == 0.0

    def test_ties_average_to_half(self) -> None:
        labels = torch.tensor([[0.0], [1.0]])
        logits = torch.tensor([[0.5], [0.5]])
        assert abs(macro_auc(logits, labels) - 0.5) < 1e-9

    def test_degenerate_classes_skipped(self) -> None:
        # class 0 evaluable; class 1 all-positive (skipped) -> mean over class 0 only.
        labels = torch.tensor([[0.0, 1.0], [1.0, 1.0]])
        logits = torch.tensor([[-1.0, 0.3], [1.0, 0.7]])
        assert macro_auc(logits, labels) == 1.0

    def test_all_degenerate_is_nan(self) -> None:
        labels = torch.ones(4, 2)
        logits = torch.randn(4, 2)
        assert macro_auc(logits, labels) != macro_auc(logits, labels)  # nan


class TestTrainStep:
    def test_one_step_lowers_loss(self, tmp_path: object) -> None:
        trainer = _trainer(tmp_path)
        ds = _SyntheticECG(16)
        ecg = ds.signals
        labels = ds.labels
        losses = [trainer.train_step(ecg, labels)[0] for _ in range(5)]
        assert losses[-1] < losses[0]

    def test_grad_norm_finite(self, tmp_path: object) -> None:
        trainer = _trainer(tmp_path)
        ds = _SyntheticECG(8)
        _, grad_norm = trainer.train_step(ds.signals, ds.labels)
        assert grad_norm == grad_norm and grad_norm >= 0.0

    def test_step_counter_advances(self, tmp_path: object) -> None:
        trainer = _trainer(tmp_path)
        ds = _SyntheticECG(8)
        trainer.train_step(ds.signals, ds.labels)
        trainer.train_step(ds.signals, ds.labels)
        assert trainer.step_idx == 2


class TestEvaluate:
    def test_returns_loss_and_auc(self, tmp_path: object) -> None:
        trainer = _trainer(tmp_path)
        loader = DataLoader(_SyntheticECG(24), batch_size=8)
        metrics = trainer.evaluate(loader)
        assert "loss" in metrics and "macro_auc" in metrics
        assert metrics["loss"] >= 0.0
        assert 0.0 <= metrics["macro_auc"] <= 1.0

    def test_auc_improves_after_training(self, tmp_path: object) -> None:
        trainer = _trainer(tmp_path)
        ds = _SyntheticECG(64)
        loader = DataLoader(ds, batch_size=16)
        before = trainer.evaluate(loader)["macro_auc"]
        for _ in range(40):
            trainer.train_step(ds.signals, ds.labels)
        after = trainer.evaluate(loader)["macro_auc"]
        # A real margin, not >= : the seeded run gains ~0.37, so a no-op or
        # near-no-op trainer (after ~= before) must not pass.
        assert after > before + 0.1


class TestCheckpointResume:
    def test_resume_restores_step_and_optimizer(self, tmp_path: object) -> None:
        trainer = _trainer(tmp_path)
        ds = _SyntheticECG(8)
        for _ in range(3):
            trainer.train_step(ds.signals, ds.labels)
        trainer.save_checkpoint()

        fresh = _trainer(tmp_path)
        assert fresh.load_checkpoint()
        assert fresh.step_idx == 3
        opt_state = fresh.optimizer.state_dict()["state"]
        assert opt_state  # optimizer momentum buffers restored

    def test_resume_round_trips_rng_state(self, tmp_path: object) -> None:
        import os

        trainer = _trainer(tmp_path)
        ds = _SyntheticECG(8)
        for _ in range(3):
            trainer.train_step(ds.signals, ds.labels)
        trainer.save_checkpoint()
        saved = torch.load(os.path.join(str(tmp_path), "trainer_state.pt"), weights_only=False)
        saved_cpu_rng = saved["rng_states"][0]["cpu"]

        torch.rand(123)  # perturb the global RNG so load must actively restore it
        fresh = _trainer(tmp_path)
        assert fresh.load_checkpoint()
        assert torch.equal(torch.get_rng_state(), saved_cpu_rng)

    def test_resume_restores_model_weights(self, tmp_path: object) -> None:
        trainer = _trainer(tmp_path)
        ds = _SyntheticECG(8)
        for _ in range(3):
            trainer.train_step(ds.signals, ds.labels)
        trainer.save_checkpoint()
        ref = trainer.evaluate(DataLoader(ds, batch_size=8))["loss"]

        fresh = _trainer(tmp_path)
        fresh.load_checkpoint()
        got = fresh.evaluate(DataLoader(ds, batch_size=8))["loss"]
        assert abs(got - ref) < 1e-5

    def test_resume_without_checkpoint_is_false(self, tmp_path: object) -> None:
        assert not _trainer(tmp_path).load_checkpoint()

    def test_old_models_pruned_keeping_two(self, tmp_path: object) -> None:
        import os

        trainer = _trainer(tmp_path)
        ds = _SyntheticECG(8)
        for _ in range(4):
            trainer.train_step(ds.signals, ds.labels)
            trainer.save_checkpoint()
        models = [f for f in os.listdir(str(tmp_path)) if f.startswith("model_step_")]
        assert len(models) <= 2

    def test_run_reaches_total_steps(self, tmp_path: object) -> None:
        trainer = _trainer(tmp_path, total_steps=6)
        loader = DataLoader(_SyntheticECG(8), batch_size=4)
        trainer.run(loader, val_loader=loader)
        assert trainer.step_idx == 6
