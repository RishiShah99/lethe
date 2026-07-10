"""Unit tests for PTBXL dataset (CPU-side, no real data required)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from lethe.medical.data import _SUPERCLASSES, PTBXL, _parse_scp_codes

_DIAG_CODES = {
    "NORM": "NORM",
    "MI": "MI",
    "STTC": "STTC",
    "CD": "CD",
    "HYP": "HYP",
    # a non-diagnostic code to confirm it is excluded
    "LNGQT": None,
}

_SCP_STMTS_ROWS = [
    # index, diagnostic, diagnostic_class, diagnostic_subclass
    ("NORM", 1, "NORM", "NORM"),
    ("IMI", 1, "MI", "IMI"),
    ("ASMI", 1, "MI", "ASMI"),
    ("STTC", 1, "STTC", "STTC"),
    ("CD", 1, "CD", "CD"),
    ("HYP", 1, "HYP", "HYP"),
    # non-diagnostic row, must be excluded from labels
    ("LNGQT", 0, "", ""),
]


def _make_scp_stmts(root: Path) -> None:
    rows = []
    for code, diag, cls_, sub in _SCP_STMTS_ROWS:
        rows.append(
            {
                "": code,
                "diagnostic": diag,
                "diagnostic_class": cls_,
                "diagnostic_subclass": sub,
            }
        )
    df = pd.DataFrame(rows).set_index("")
    df.to_csv(root / "scp_statements.csv")


def _scp_str(d: dict[str, float]) -> str:
    return str(d)


def _make_database(root: Path, n_per_fold: int = 3) -> pd.DataFrame:
    """Produce a minimal ptbxl_database.csv with records across all 10 folds."""
    records = []
    ecg_id = 1
    for fold in range(1, 11):
        for i in range(n_per_fold):
            subdir = f"00{ecg_id:03d}"
            fname_lr = f"records100/{subdir}/{ecg_id:05d}_lr"
            fname_hr = f"records500/{subdir}/{ecg_id:05d}_hr"
            # vary scp_codes per record
            if i == 0:
                scp = _scp_str({"NORM": 100.0})
            elif i == 1:
                scp = _scp_str({"IMI": 80.0, "CD": 20.0})
            else:
                scp = _scp_str({"LNGQT": 100.0})  # non-diagnostic only
            records.append(
                {
                    "ecg_id": ecg_id,
                    "filename_lr": fname_lr,
                    "filename_hr": fname_hr,
                    "strat_fold": fold,
                    "scp_codes": scp,
                }
            )
            ecg_id += 1
    df = pd.DataFrame(records).set_index("ecg_id")
    df.to_csv(root / "ptbxl_database.csv")
    return df


def _make_fixture(root: Path) -> None:
    _make_scp_stmts(root)
    _make_database(root)
    # Create stub WFDB header/data files (not actually read; wfdb is mocked)
    for rate in ("records100", "records500"):
        (root / rate).mkdir(parents=True, exist_ok=True)


@pytest.fixture()
def ptbxl_root(tmp_path: Path) -> Path:
    _make_fixture(tmp_path)
    return tmp_path


def _mock_rdsamp(record_path: str, **_: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a (1000, 12) float64 array and empty fields dict."""
    signal = np.zeros((1000, 12), dtype=np.float64)
    return signal, {}


class TestParseSCPCodes:
    def test_normal_dict(self) -> None:
        raw = "{'NORM': 100.0, 'IMI': 80.0}"
        result = _parse_scp_codes(raw)
        assert result == {"NORM": 100.0, "IMI": 80.0}

    def test_int_likelihood(self) -> None:
        raw = "{'NORM': 100, 'IMI': 0}"
        result = _parse_scp_codes(raw)
        assert result == {"NORM": 100.0, "IMI": 0.0}
        for v in result.values():
            assert isinstance(v, float)

    def test_empty_dict(self) -> None:
        assert _parse_scp_codes("{}") == {}

    def test_non_dict_returns_empty(self) -> None:
        # Defensive: column can be malformed in edge cases
        assert _parse_scp_codes("[]") == {}


class TestPTBXLConstruction:
    def test_missing_database_csv(self, tmp_path: Path) -> None:
        _make_scp_stmts(tmp_path)
        with pytest.raises(FileNotFoundError):
            PTBXL(tmp_path)

    def test_missing_scp_statements(self, tmp_path: Path) -> None:
        _make_database(tmp_path)
        with pytest.raises(FileNotFoundError):
            PTBXL(tmp_path)

    def test_invalid_sampling_rate(self, ptbxl_root: Path) -> None:
        with pytest.raises(ValueError, match="sampling_rate"):
            PTBXL(ptbxl_root, sampling_rate=250)  # type: ignore[arg-type]

    def test_invalid_split(self, ptbxl_root: Path) -> None:
        with pytest.raises(ValueError, match="split"):
            PTBXL(ptbxl_root, split="holdout")  # type: ignore[arg-type]

    def test_invalid_label_set(self, ptbxl_root: Path) -> None:
        with pytest.raises(ValueError, match="label_set"):
            PTBXL(ptbxl_root, label_set="binary")  # type: ignore[arg-type]

    def test_train_split_length(self, ptbxl_root: Path) -> None:
        # folds 1-8, 3 records/fold = 24
        ds = PTBXL(ptbxl_root, split="train")
        assert len(ds) == 24

    def test_val_split_length(self, ptbxl_root: Path) -> None:
        ds = PTBXL(ptbxl_root, split="val")
        assert len(ds) == 3

    def test_test_split_length(self, ptbxl_root: Path) -> None:
        ds = PTBXL(ptbxl_root, split="test")
        assert len(ds) == 3

    def test_superclass_n_classes(self, ptbxl_root: Path) -> None:
        ds = PTBXL(ptbxl_root, label_set="superclass")
        assert ds.n_classes == 5
        assert ds.classes == _SUPERCLASSES

    def test_subclass_n_classes(self, ptbxl_root: Path) -> None:
        # Diagnostic codes in the fixture: NORM, IMI, ASMI, STTC, CD, HYP
        ds = PTBXL(ptbxl_root, label_set="subclass")
        assert ds.n_classes == 6
        assert "IMI" in ds.classes
        assert "ASMI" in ds.classes
        assert "LNGQT" not in ds.classes  # non-diagnostic excluded

    def test_classes_property_returns_copy(self, ptbxl_root: Path) -> None:
        ds = PTBXL(ptbxl_root)
        c1 = ds.classes
        c1.append("EXTRA")
        assert "EXTRA" not in ds.classes


class TestPTBXLLabels:
    """Label vector tests do not require wfdb; they inspect _labels directly."""

    def test_norm_superclass_label(self, ptbxl_root: Path) -> None:
        ds = PTBXL(ptbxl_root, split="train", label_set="superclass")
        # First record per fold has scp={'NORM': 100.0}
        label = ds._labels[0]
        assert label.dtype == torch.float32
        assert label.shape == (5,)
        norm_idx = ds._class_index["NORM"]
        assert label[norm_idx].item() == 1.0
        # All other classes must be 0
        others = label.clone()
        others[norm_idx] = 0.0
        assert others.sum().item() == 0.0

    def test_mi_cd_multi_hot(self, ptbxl_root: Path) -> None:
        # Second record per fold: {'IMI': 80.0, 'CD': 20.0}; IMI maps to MI superclass, CD to CD
        ds = PTBXL(ptbxl_root, split="train", label_set="superclass")
        label = ds._labels[1]
        mi_idx = ds._class_index["MI"]
        cd_idx = ds._class_index["CD"]
        assert label[mi_idx].item() == 1.0
        assert label[cd_idx].item() == 1.0
        # NORM, STTC, HYP must be 0
        for cls in ["NORM", "STTC", "HYP"]:
            assert label[ds._class_index[cls]].item() == 0.0

    def test_non_diagnostic_code_excluded(self, ptbxl_root: Path) -> None:
        # Third record per fold: {'LNGQT': 100.0}, LNGQT is non-diagnostic
        ds = PTBXL(ptbxl_root, split="train", label_set="superclass")
        label = ds._labels[2]
        assert label.sum().item() == 0.0

    def test_subclass_label_imi(self, ptbxl_root: Path) -> None:
        ds = PTBXL(ptbxl_root, split="train", label_set="subclass")
        label = ds._labels[1]  # IMI: 80.0, CD: 20.0
        imi_idx = ds._class_index["IMI"]
        cd_idx = ds._class_index["CD"]
        assert label[imi_idx].item() == 1.0
        assert label[cd_idx].item() == 1.0

    def test_min_likelihood_filters_low_codes(self, ptbxl_root: Path) -> None:
        # IMI=80.0, CD=20.0; with min_likelihood=50.0 only IMI passes
        ds = PTBXL(
            ptbxl_root,
            split="train",
            label_set="superclass",
            min_likelihood=50.0,
        )
        label = ds._labels[1]
        mi_idx = ds._class_index["MI"]
        cd_idx = ds._class_index["CD"]
        assert label[mi_idx].item() == 1.0
        assert label[cd_idx].item() == 0.0

    def test_min_likelihood_zero_includes_all(self, ptbxl_root: Path) -> None:
        ds = PTBXL(ptbxl_root, split="train", label_set="superclass", min_likelihood=0.0)
        label = ds._labels[1]
        assert label[ds._class_index["MI"]].item() == 1.0
        assert label[ds._class_index["CD"]].item() == 1.0

    def test_min_likelihood_100_strict(self, ptbxl_root: Path) -> None:
        # Only NORM (100.0) passes; IMI (80.0) and CD (20.0) do not
        ds = PTBXL(ptbxl_root, split="train", label_set="superclass", min_likelihood=100.0)
        label_norm = ds._labels[0]
        label_imi_cd = ds._labels[1]
        assert label_norm[ds._class_index["NORM"]].item() == 1.0
        assert label_imi_cd.sum().item() == 0.0

    def test_label_is_float32(self, ptbxl_root: Path) -> None:
        ds = PTBXL(ptbxl_root)
        assert ds._labels[0].dtype == torch.float32

    def test_all_labels_correct_shape(self, ptbxl_root: Path) -> None:
        ds = PTBXL(ptbxl_root, split="train")
        for lbl in ds._labels:
            assert lbl.shape == (ds.n_classes,)


class TestPTBXLGetItem:
    def test_signal_shape_100hz(self, ptbxl_root: Path) -> None:
        with patch("lethe.medical.data.wfdb") as mock_wfdb:
            mock_wfdb.rdsamp.return_value = (np.zeros((1000, 12), dtype=np.float64), {})
            ds = PTBXL(ptbxl_root, sampling_rate=100)
            sig, _ = ds[0]
        assert sig.shape == (12, 1000)
        assert sig.dtype == torch.float32

    def test_signal_shape_500hz(self, ptbxl_root: Path) -> None:
        with patch("lethe.medical.data.wfdb") as mock_wfdb:
            mock_wfdb.rdsamp.return_value = (np.zeros((5000, 12), dtype=np.float64), {})
            ds = PTBXL(ptbxl_root, sampling_rate=500)
            sig, _ = ds[0]
        assert sig.shape == (12, 5000)

    def test_signal_is_float32(self, ptbxl_root: Path) -> None:
        with patch("lethe.medical.data.wfdb") as mock_wfdb:
            mock_wfdb.rdsamp.return_value = (np.ones((1000, 12), dtype=np.float64), {})
            ds = PTBXL(ptbxl_root, sampling_rate=100)
            sig, _ = ds[0]
        assert sig.dtype == torch.float32

    def test_label_shape_matches_n_classes(self, ptbxl_root: Path) -> None:
        with patch("lethe.medical.data.wfdb") as mock_wfdb:
            mock_wfdb.rdsamp.return_value = (np.zeros((1000, 12), dtype=np.float64), {})
            ds = PTBXL(ptbxl_root, sampling_rate=100, label_set="superclass")
            _, lbl = ds[0]
        assert lbl.shape == (5,)

    def test_wfdb_called_with_correct_path(self, ptbxl_root: Path) -> None:
        with patch("lethe.medical.data.wfdb") as mock_wfdb:
            mock_wfdb.rdsamp.return_value = (np.zeros((1000, 12), dtype=np.float64), {})
            ds = PTBXL(ptbxl_root, sampling_rate=100, split="test")
            _ = ds[0]
        call_path = mock_wfdb.rdsamp.call_args[0][0]
        assert "records100" in call_path
        assert str(ptbxl_root) in call_path

    def test_500hz_uses_filename_hr(self, ptbxl_root: Path) -> None:
        with patch("lethe.medical.data.wfdb") as mock_wfdb:
            mock_wfdb.rdsamp.return_value = (np.zeros((5000, 12), dtype=np.float64), {})
            ds = PTBXL(ptbxl_root, sampling_rate=500, split="test")
            _ = ds[0]
        call_path = mock_wfdb.rdsamp.call_args[0][0]
        assert "records500" in call_path

    def test_signal_normalized_per_lead(self, ptbxl_root: Path) -> None:
        # The loader NaN-guards then per-lead z-scores; raw mV scales would otherwise NaN the deep SSM.
        arr = np.arange(12000, dtype=np.float64).reshape(1000, 12)
        with patch("lethe.medical.data.wfdb") as mock_wfdb:
            mock_wfdb.rdsamp.return_value = (arr, {})
            ds = PTBXL(ptbxl_root, sampling_rate=100, split="test")
            sig, _ = ds[0]
        assert sig.shape == (12, 1000)
        for lead in range(12):
            x = torch.from_numpy(arr[:, lead].astype(np.float32))
            expected = (x - x.mean()) / x.std().clamp_min(1e-6)
            torch.testing.assert_close(sig[lead], expected, rtol=1e-4, atol=1e-4)

    def test_nonfinite_samples_guarded(self, ptbxl_root: Path) -> None:
        arr = np.arange(12000, dtype=np.float64).reshape(1000, 12)
        arr[0, 0] = np.nan
        arr[1, 1] = np.inf
        with patch("lethe.medical.data.wfdb") as mock_wfdb:
            mock_wfdb.rdsamp.return_value = (arr, {})
            ds = PTBXL(ptbxl_root, sampling_rate=100, split="test")
            sig, _ = ds[0]
        assert torch.isfinite(sig).all()

    def test_getitem_returns_tuple(self, ptbxl_root: Path) -> None:
        with patch("lethe.medical.data.wfdb") as mock_wfdb:
            mock_wfdb.rdsamp.return_value = (np.zeros((1000, 12), dtype=np.float64), {})
            ds = PTBXL(ptbxl_root, sampling_rate=100)
            item = ds[0]
        assert isinstance(item, tuple)
        assert len(item) == 2


class TestSamplingRateRouting:
    def test_100hz_filename_col(self, ptbxl_root: Path) -> None:
        # All records at 100 Hz must reference records100/
        ds = PTBXL(ptbxl_root, sampling_rate=100, split="train")
        for path in ds._records:
            assert path.startswith("records100/")

    def test_500hz_filename_col(self, ptbxl_root: Path) -> None:
        ds = PTBXL(ptbxl_root, sampling_rate=500, split="train")
        for path in ds._records:
            assert path.startswith("records500/")
