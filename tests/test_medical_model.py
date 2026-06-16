"""CPU tests for Mamba3ECGClassifier (Phase F.2).

tiny() forwards on CPU (eager fallback); analytic param count matches the live
module and the b1 config is ~1.1B.
"""

from __future__ import annotations

import torch

from flash_mamba_rl.medical.model import Mamba3Config, Mamba3ECGClassifier


class TestForward:
    def test_output_shape(self) -> None:
        cfg = Mamba3Config.tiny()
        model = Mamba3ECGClassifier(cfg)
        out = model(torch.randn(3, 12, 64))
        assert out.shape == (3, cfg.n_classes)

    def test_custom_n_classes(self) -> None:
        cfg = Mamba3Config(d_model=32, n_layers=2, d_state=8, chunk_size=8, n_classes=23)
        model = Mamba3ECGClassifier(cfg)
        assert model(torch.randn(2, 12, 64)).shape == (2, 23)

    def test_logits_finite(self) -> None:
        model = Mamba3ECGClassifier(Mamba3Config.tiny())
        out = model(torch.randn(2, 12, 64))
        assert torch.isfinite(out).all()


class TestParamCount:
    def test_analytic_matches_live_tiny(self) -> None:
        cfg = Mamba3Config.tiny()
        model = Mamba3ECGClassifier(cfg)
        assert model.param_count() == cfg.analytic_param_count()

    def test_b1_is_about_1_1b(self) -> None:
        n = Mamba3Config.b1().analytic_param_count()
        assert 1.05e9 < n < 1.15e9

    def test_analytic_matches_live_b1_scaled(self) -> None:
        # Verify the formula on a mid config without building the 1.1B model.
        cfg = Mamba3Config(d_model=256, n_layers=4, d_state=32, chunk_size=8)
        model = Mamba3ECGClassifier(cfg)
        assert model.param_count() == cfg.analytic_param_count()
