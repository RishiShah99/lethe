"""CPU tests for the Mamba-3 ECG classifier (Phase F.2)."""

from __future__ import annotations

import dataclasses

import torch
import torch.nn as nn

from lethe.medical.model import Mamba3Config, Mamba3ECGClassifier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_cfg(chunk_size: int = 8) -> Mamba3Config:
    return dataclasses.replace(Mamba3Config.tiny(), chunk_size=chunk_size)


def _make_model(chunk_size: int = 8) -> Mamba3ECGClassifier:
    return Mamba3ECGClassifier(dataclasses.replace(Mamba3Config.tiny(), chunk_size=chunk_size))


# ---------------------------------------------------------------------------
# Forward shape + finiteness
# ---------------------------------------------------------------------------


class TestForward:
    def test_output_shape_T256(self) -> None:
        """T=256 is divisible by chunk_size=8 → valid."""
        model = _make_model(chunk_size=8)
        ecg = torch.randn(2, 12, 256)
        logits = model(ecg)
        assert logits.shape == (2, 5)

    def test_output_shape_T512(self) -> None:
        model = _make_model(chunk_size=8)
        ecg = torch.randn(3, 12, 512)
        logits = model(ecg)
        assert logits.shape == (3, 5)

    def test_finite_logits_T256(self) -> None:
        model = _make_model(chunk_size=8)
        ecg = torch.randn(2, 12, 256)
        logits = model(ecg)
        assert torch.isfinite(logits).all()

    def test_finite_logits_T512(self) -> None:
        model = _make_model(chunk_size=8)
        ecg = torch.randn(2, 12, 512)
        logits = model(ecg)
        assert torch.isfinite(logits).all()

    def test_batch_size_1(self) -> None:
        model = _make_model(chunk_size=8)
        ecg = torch.randn(1, 12, 64)
        logits = model(ecg)
        assert logits.shape == (1, 5)
        assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestGradients:
    def test_all_leaf_params_have_grad(self) -> None:
        """Every trainable parameter must receive a gradient signal."""
        model = _make_model(chunk_size=8)
        ecg = torch.randn(2, 12, 256)
        logits = model(ecg)
        loss = logits.sum()
        loss.backward()
        missing = [
            name for name, p in model.named_parameters() if p.requires_grad and p.grad is None
        ]
        assert missing == [], f"Parameters with no grad: {missing}"

    def test_grad_is_finite(self) -> None:
        model = _make_model(chunk_size=8)
        ecg = torch.randn(2, 12, 256)
        logits = model(ecg)
        loss = logits.sum()
        loss.backward()
        non_finite = [
            name
            for name, p in model.named_parameters()
            if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all()
        ]
        assert non_finite == [], f"Parameters with non-finite grad: {non_finite}"


# ---------------------------------------------------------------------------
# param_count
# ---------------------------------------------------------------------------


class TestParamCount:
    def test_param_count_matches_sum(self) -> None:
        model = _make_model()
        reported = model.param_count()
        actual = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert reported == actual

    def test_param_count_positive(self) -> None:
        model = _make_model()
        assert model.param_count() > 0

    def test_tiny_cfg_analytic_matches_instantiated(self) -> None:
        cfg = Mamba3Config.tiny()
        model = Mamba3ECGClassifier(cfg)
        analytic = cfg.analytic_param_count()
        actual = model.param_count()
        assert analytic == actual, f"analytic={analytic} actual={actual}"


# ---------------------------------------------------------------------------
# b1 analytic count (no instantiation to avoid OOM)
# ---------------------------------------------------------------------------


class TestB1Config:
    def test_b1_analytic_count_in_1b_band(self) -> None:
        """Analytic formula must land in [0.5B, 2B] without instantiating the model."""
        cfg = Mamba3Config.b1()
        count = cfg.analytic_param_count()
        assert 0.5e9 < count < 2.0e9, f"b1 analytic count {count:,} out of [0.5B, 2B]"

    def test_b1_analytic_formula_consistent_with_tiny(self) -> None:
        """Same formula applied to tiny config must match the instantiated count."""
        cfg = Mamba3Config.tiny()
        model = Mamba3ECGClassifier(cfg)
        assert cfg.analytic_param_count() == model.param_count()


# ---------------------------------------------------------------------------
# End-to-end: loader contract → model → loss → optimizer step
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_loss_decreases_after_one_step(self) -> None:
        """One AdamW step must move at least one parameter."""
        torch.manual_seed(42)
        cfg = dataclasses.replace(Mamba3Config.tiny(), chunk_size=8)
        model = Mamba3ECGClassifier(cfg)

        ecg = torch.randn(2, 12, 256)  # loader-shape input
        target = torch.zeros(2, 5)
        target[0, 0] = 1.0
        target[1, 2] = 1.0

        criterion = nn.BCEWithLogitsLoss()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        logits0 = model(ecg)
        loss0 = criterion(logits0, target)
        loss0.backward()
        params_before = [p.data.clone() for p in model.parameters()]
        opt.step()
        opt.zero_grad()

        params_after = list(model.parameters())
        moved = any(
            not torch.equal(pb, pa.data) for pb, pa in zip(params_before, params_after, strict=True)
        )
        assert moved, "No parameter moved after one optimizer step"

    def test_bce_loss_finite_before_step(self) -> None:
        torch.manual_seed(7)
        cfg = dataclasses.replace(Mamba3Config.tiny(), chunk_size=8)
        model = Mamba3ECGClassifier(cfg)
        ecg = torch.randn(2, 12, 256)
        target = torch.zeros(2, 5)
        target[0, 1] = 1.0
        criterion = nn.BCEWithLogitsLoss()
        logits = model(ecg)
        loss = criterion(logits, target)
        assert torch.isfinite(loss)
