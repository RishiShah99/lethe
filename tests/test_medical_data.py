"""CPU tests for PTBXL label-building (Phase F.1).

Synthetic ptbxl_database.csv + scp_statements.csv in tmp_path — no real WFDB
records, so __getitem__ (wfdb.rdsamp) is never called; only the index/label
construction is exercised.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from flash_mamba_rl.medical.data import PTBXL

_SCP = pd.DataFrame(
    {
        "diagnostic": [1, 1, 1, 1, 1, 0],
        "diagnostic_class": ["NORM", "MI", "STTC", "CD", "HYP", "NORM"],
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
    def test_subclass_vocab_is_diagnostic_codes(self, root: Path) -> None:
        ds = PTBXL(root, sampling_rate=100, split="train", label_set="subclass")
        assert ds.n_classes == 5  # NORM, IMI, NDT, IRBBB, LVH (diagnostic==1)
        assert "SR" not in ds.classes  # diagnostic==0 excluded


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
