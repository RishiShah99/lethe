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


class TestStability:
    """A = -exp(A_log) must keep the scan a strict contraction for any A_log.

    The prior init passed log_A directly as A, leaving the j=0 column at A=0 (a
    pure integrator) and letting A drift positive under training → the L=1000
    scan overflowed to NaN. These pin that the parametrisation now precludes it.
    """

    def test_effective_A_strictly_negative_at_init(self) -> None:
        model = Mamba3ECGClassifier(Mamba3Config.tiny())
        for block in model.blocks:
            A = -torch.exp(block.A_log)
            assert (A < 0).all()

    def test_forward_finite_when_A_log_drifts_positive(self) -> None:
        # Simulate a positive drift that the old A=log_A init would have turned
        # into a_bar>1 (geometric blow-up); A=-exp(A_log) clamps it to (0,1).
        model = Mamba3ECGClassifier(Mamba3Config.tiny())
        with torch.no_grad():
            for block in model.blocks:
                block.A_log.fill_(5.0)
        out = model(torch.randn(2, 12, 256) * 6.0)
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


class TestRegularization:
    def test_dropout_p0_is_identity(self) -> None:
        # Default dropout 0.0 -> train and eval forwards agree (no params added).
        model = Mamba3ECGClassifier(Mamba3Config.tiny())
        ecg = torch.randn(2, 12, 64)
        model.train()
        a = model(ecg)
        model.eval()
        b = model(ecg)
        assert torch.allclose(a, b, atol=1e-6)
        assert model.param_count() == Mamba3Config.tiny().analytic_param_count()

    def test_dropout_active_in_train_mode(self) -> None:
        cfg = Mamba3Config(d_model=32, n_layers=2, d_state=8, chunk_size=8, dropout=0.5)
        model = Mamba3ECGClassifier(cfg)
        ecg = torch.randn(2, 12, 64)
        model.train()
        torch.manual_seed(0)
        a = model(ecg)
        torch.manual_seed(1)
        b = model(ecg)
        # Different dropout masks -> different train outputs; eval is deterministic.
        assert not torch.allclose(a, b)
        model.eval()
        assert torch.allclose(model(ecg), model(ecg))

    def test_b_mid_param_count_in_range(self) -> None:
        n = Mamba3Config.b_mid().analytic_param_count()
        assert 1.0e8 < n < 2.5e8  # ~160M, well under b1's 1.1B
