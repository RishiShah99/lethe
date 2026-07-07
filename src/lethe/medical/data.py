"""PTB-XL ECG dataset loader (CPU-side).

Expected on-disk layout (standard PhysioNet download, do not alter):
    <root>/
        ptbxl_database.csv        -- 21 799 record index, one row per ECG
        scp_statements.csv        -- SCP-ECG code taxonomy (index = code string)
        records100/<subdir>/*.hea  -- 100 Hz WFDB records (filename_lr column)
        records500/<subdir>/*.hea  -- 500 Hz WFDB records (filename_hr column)

Fetch command (box side):
    wget -r -N -c -np \
        https://physionet.org/files/ptb-xl/1.0.3/ \
        -P <root>

Splits follow the official strat_fold protocol from the PTB-XL paper
(Strodthoff et al. 2020):  train = folds 1-8 / val = fold 9 / test = fold 10.

Label sets:
  "superclass" -- 5 diagnostic superclasses: NORM / MI / STTC / CD / HYP
                  (diagnostic_class column of scp_statements, rows where
                   diagnostic == 1).
  "subclass"   -- 23 diagnostic subclasses: each diagnostic SCP code (rows
                  where diagnostic == 1) rolled up into its diagnostic_subclass
                  grouping — the PTB-XL benchmark's 23-way diagnostic task.

min_likelihood:
  The scp_codes column is a dict {code: likelihood}.  A code is included in
  the label vector when likelihood >= min_likelihood.  0.0 includes every
  annotated code; 100.0 restricts to codes annotated with full certainty.
  The published PTB-XL benchmarks use 0.0 (all annotated codes).

Output of __getitem__:
  signal : torch.Tensor  shape (12, T), float32
      T = sampling_rate * recording_duration (T=1000 at 100 Hz, T=5000 at 500 Hz)
  label  : torch.Tensor  shape (C,), float32 multi-hot
      C = 5 for superclass, 23 for subclass (diagnostic_subclass groups)
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset

__all__ = ["PTBXL"]

_SPLIT_FOLDS: dict[str, list[int]] = {
    "train": list(range(1, 9)),
    "val": [9],
    "test": [10],
}

_SUPERCLASSES: list[str] = ["NORM", "MI", "STTC", "CD", "HYP"]


def _parse_scp_codes(raw: str) -> dict[str, float]:
    """ast.literal_eval the scp_codes column; always returns a dict."""
    parsed = ast.literal_eval(raw)
    if not isinstance(parsed, dict):
        return {}
    return {str(k): float(v) for k, v in parsed.items()}


class PTBXL(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """torch Dataset over a local PTB-XL root directory.

    Parameters
    ----------
    root:
        Path to the directory that contains ``ptbxl_database.csv`` and
        ``scp_statements.csv``.
    sampling_rate:
        100  → records100/ (filename_lr column, T = 1 000 samples)
        500  → records500/ (filename_hr column, T = 5 000 samples)
    split:
        "train" | "val" | "test" — official strat_fold partition.
    label_set:
        "superclass" → 5-class multi-hot (NORM/MI/STTC/CD/HYP)
        "subclass"   → 23-class diagnostic-subclass multi-hot
    min_likelihood:
        Include an SCP code in the label when its annotated likelihood is
        >= this value.  0.0 = all codes annotated for the record.

    Raises
    ------
    FileNotFoundError
        If ``ptbxl_database.csv`` or ``scp_statements.csv`` are absent
        under *root* — caller must fetch the dataset first.
    ValueError
        If *sampling_rate*, *split*, or *label_set* are invalid.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        sampling_rate: Literal[100, 500] = 500,
        split: Literal["train", "val", "test"] = "train",
        label_set: Literal["superclass", "subclass"] = "superclass",
        min_likelihood: float = 0.0,
        augment: bool = False,
    ) -> None:
        if sampling_rate not in (100, 500):
            raise ValueError(f"sampling_rate must be 100 or 500, got {sampling_rate}")
        if split not in _SPLIT_FOLDS:
            raise ValueError(f"split must be one of {list(_SPLIT_FOLDS)}, got {split!r}")
        if label_set not in ("superclass", "subclass"):
            raise ValueError(f"label_set must be 'superclass' or 'subclass', got {label_set!r}")

        self._root = Path(root)
        self._sampling_rate = sampling_rate
        self._split = split
        self._label_set = label_set
        self._min_likelihood = min_likelihood
        # #19: train-only augmentation against overfit. Off for val/test so the
        # eval signal is the clean record. Applied after the per-lead z-score.
        self._augment = augment

        db_path = self._root / "ptbxl_database.csv"
        scp_path = self._root / "scp_statements.csv"
        if not db_path.exists():
            raise FileNotFoundError(db_path)
        if not scp_path.exists():
            raise FileNotFoundError(scp_path)

        db = pd.read_csv(db_path, index_col="ecg_id")
        scp_stmts = pd.read_csv(scp_path, index_col=0)

        # Restrict to diagnostic codes only.
        diag_stmts = scp_stmts[scp_stmts["diagnostic"] == 1].copy()

        # Build the ordered label vocabulary.
        if label_set == "superclass":
            self._classes: list[str] = _SUPERCLASSES
            # code → superclass string
            self._code_to_label: dict[str, str] = {
                idx: str(row["diagnostic_class"])
                for idx, row in diag_stmts.iterrows()
                if str(row["diagnostic_class"]) in _SUPERCLASSES
            }
        else:
            # PTB-XL diagnostic-subclass task: each diagnostic SCP code rolls up
            # into its diagnostic_subclass grouping (23 subclasses in PTB-XL
            # 1.0.3), the unit the published benchmark reports — not the raw
            # per-code set.
            sub_col = diag_stmts["diagnostic_subclass"].dropna().astype(str)
            sub_col = sub_col[sub_col.str.len() > 0]
            code_to_sub: dict[str, str] = {str(code): str(sub) for code, sub in sub_col.items()}
            self._classes = sorted(set(code_to_sub.values()))
            self._code_to_label = code_to_sub

        self._class_index: dict[str, int] = {c: i for i, c in enumerate(self._classes)}
        self._n_classes = len(self._classes)

        # Split filtering.
        folds = _SPLIT_FOLDS[split]
        mask = db["strat_fold"].isin(folds)
        split_db = db[mask].copy()

        # Filename column.
        fname_col = "filename_lr" if sampling_rate == 100 else "filename_hr"

        self._records: list[str] = split_db[fname_col].tolist()
        self._labels: list[torch.Tensor] = [
            self._build_label(_parse_scp_codes(raw)) for raw in split_db["scp_codes"].tolist()
        ]

    def _build_label(self, scp_codes: dict[str, float]) -> torch.Tensor:
        vec = torch.zeros(self._n_classes, dtype=torch.float32)
        for code, likelihood in scp_codes.items():
            if likelihood < self._min_likelihood:
                continue
            label = self._code_to_label.get(code)
            if label is None:
                continue
            idx = self._class_index.get(label)
            if idx is not None:
                vec[idx] = 1.0
        return vec

    @property
    def classes(self) -> list[str]:
        """Ordered label vocabulary (length == label vector size)."""
        return list(self._classes)

    @property
    def n_classes(self) -> int:
        """Number of label classes."""
        return self._n_classes

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(signal, label)``.

        signal : float32, shape ``(12, T)``
        label  : float32 multi-hot, shape ``(C,)``
        """
        record_path = str(self._root / self._records[idx])
        signal, _ = wfdb.rdsamp(record_path)
        # wfdb returns (T, 12); transpose to (12, T).
        sig_t = torch.from_numpy(np.array(signal, dtype=np.float32)).T
        # Some PTB-XL records carry NaN/Inf samples; raw mV scales also vary per
        # lead. Both NaN the deep SSM forward, so guard then per-lead z-score
        # (zero mean / unit std over time; flat leads -> zero via the clamp).
        sig_t = torch.nan_to_num(sig_t, nan=0.0, posinf=0.0, neginf=0.0)
        mean = sig_t.mean(dim=1, keepdim=True)
        std = sig_t.std(dim=1, keepdim=True).clamp_min(1e-6)
        sig_t = (sig_t - mean) / std
        if self._augment:
            sig_t = self._augment_signal(sig_t)
        return sig_t, self._labels[idx]

    @staticmethod
    def _augment_signal(sig: torch.Tensor) -> torch.Tensor:
        """Light ECG augmentation on a z-scored (12, T) signal (train only).

        Composes per-signal amplitude jitter, additive Gaussian noise, a circular
        time shift, and occasional lead masking — perturbations that preserve the
        diagnostic morphology while breaking memorization of exact waveforms.
        Uses the global torch RNG, which DataLoader seeds per worker.
        """
        n_leads, t = sig.shape
        sig = sig * (1.0 + 0.1 * (torch.rand(n_leads, 1) - 0.5) * 2.0)  # +/-10% per-lead scale
        sig = sig + 0.08 * torch.randn_like(sig)  # additive noise
        shift = int(torch.randint(-t // 50, t // 50 + 1, (1,)).item())  # +/-2% circular shift
        if shift:
            sig = torch.roll(sig, shifts=shift, dims=1)
        if float(torch.rand(1).item()) < 0.3:  # drop one lead 30% of the time
            sig[int(torch.randint(0, n_leads, (1,)).item())] = 0.0
        return sig
