"""CPU tests for PTBXL label-building (Phase F.1)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from lethe.medical.data import PTBXL

_SCP = pd.DataFrame(
    {
        "diagnostic": [1, 1, 1, 1, 1, 0],
        "diagnostic_class": ["NORM", "MI", "STTC", "CD", "HYP", "NORM"],
        # NDT rolls up to STTC (differs from raw code), proving the subclass column is read.
        "diagnostic_subclass": ["NORM", "IMI", "STTC", "IRBBB", "LVH", "SR"],
    },
    index=["NORM", "IMI", "NDT", "IRBBB", "LVH", "SR"],
)


def _write_db(root: Path, rows: list[dict[str, object]]) -> None:
    _SCP.to_csv(root / "scp_statements.csv")
    pd.DataFrame(rows).set_index("ecg_id").to_csv(root / "ptbxl_database.csv")


def _row(ecg_id: int, fold: int, scp: str) -> dict[str, object]:
    return {
        "ecg_id": ecg_id,
        "strat_fold": fold,
        "scp_codes": scp,
        "filename_lr": f"records100/{ecg_id}_lr",
        "filename_hr": f"records500/{ecg_id}_hr",
    }


@pytest.fixture
def root(tmp_path: Path) -> Path:
    _write_db(
        tmp_path,
        [
            _row(1, 1, "{'NORM': 100.0}"),
            _row(2, 1, "{'IMI': 80.0, 'NDT': 15.0}"),
            _row(3, 9, "{'LVH': 100.0}"),
            _row(4, 10, "{'IRBBB': 0.0}"),
        ],
    )
    return tmp_path


class TestSuperclass:
    def test_n_classes_is_five(self, root: Path) -> None:
        ds = PTBXL(root, sampling_rate=100, split="train", label_set="superclass")
        assert ds.n_classes == 5
        assert ds.classes == ["NORM", "MI", "STTC", "CD", "HYP"]

    def test_train_split_size(self, root: Path) -> None:
        ds = PTBXL(root, sampling_rate=100, split="train", label_set="superclass")
        assert len(ds) == 2  # folds 1-8

    def test_label_maps_codes_to_superclass(self, root: Path) -> None:
        ds = PTBXL(root, sampling_rate=100, split="train", label_set="superclass")
        # ecg_id 2: IMI->MI, NDT->STTC
        _, label = ds._records[1], ds._labels[1]
        assert label[ds._class_index["MI"]] == 1.0
        assert label[ds._class_index["STTC"]] == 1.0
        assert label[ds._class_index["NORM"]] == 0.0

    def test_min_likelihood_filters_codes(self, root: Path) -> None:
        ds = PTBXL(
            root, sampling_rate=100, split="train", label_set="superclass", min_likelihood=50.0
        )
        # ecg_id 2: IMI@80 kept, NDT@15 dropped
        label = ds._labels[1]
        assert label[ds._class_index["MI"]] == 1.0
        assert label[ds._class_index["STTC"]] == 0.0


class TestSubclass:
    def test_subclass_vocab_is_diagnostic_subclasses(self, root: Path) -> None:
        ds = PTBXL(root, sampling_rate=100, split="train", label_set="subclass")
        # 5 diagnostic codes → subclasses {NORM, IMI, STTC, IRBBB, LVH}
        assert ds.n_classes == 5
        assert "STTC" in ds.classes  # NDT's diagnostic_subclass, not the raw code
        assert "NDT" not in ds.classes  # raw code names are not the vocabulary
        assert "SR" not in ds.classes  # diagnostic==0 excluded

    def test_subclass_label_rolls_up_to_subclass(self, root: Path) -> None:
        ds = PTBXL(root, sampling_rate=100, split="train", label_set="subclass")
        # ecg_id 2: IMI stays IMI, NDT rolls to STTC; both land as positive labels.
        label = ds._labels[1]
        assert label[ds._class_index["IMI"]] == 1.0
        assert label[ds._class_index["STTC"]] == 1.0


class TestValidation:
    def test_val_and_test_folds(self, root: Path) -> None:
        val = PTBXL(root, sampling_rate=100, split="val", label_set="superclass")
        test = PTBXL(root, sampling_rate=100, split="test", label_set="superclass")
        assert len(val) == 1 and len(test) == 1

    def test_bad_sampling_rate(self, root: Path) -> None:
        with pytest.raises(ValueError):
            PTBXL(root, sampling_rate=250)  # type: ignore[arg-type]

    def test_missing_files(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            PTBXL(tmp_path, sampling_rate=100)


class TestAugment:
    """_augment_signal is a pure tensor op, testable without WFDB records."""

    def test_preserves_shape_and_finite(self) -> None:
        import torch

        torch.manual_seed(0)
        sig = torch.randn(12, 1000)
        out = PTBXL._augment_signal(sig)
        assert out.shape == sig.shape
        assert torch.isfinite(out).all()

    def test_changes_signal(self) -> None:
        import torch

        torch.manual_seed(0)
        sig = torch.randn(12, 1000)
        assert not torch.allclose(PTBXL._augment_signal(sig), sig)
