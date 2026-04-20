"""
FraudHandle — DatasetHandle implementation for bb_datasets fraud data.

Wraps the DuckDB-backed bb_datasets transaction data for use with
the air-lab-os engine. The engine sees only DatasetHandle — it never
imports from bb_datasets directly.

Split: first 80% by txn_id sort order = train, last 20% = eval.
Split is fixed and deterministic. Do not parameterise it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from functools import cached_property
import importlib

import pandas as pd

_BB_PATH = Path(__file__).parent.parent.parent.parent / "bb_datasets"
from datasets.base import DatasetHandle, DatasetMeta


def _import_bb_fraud_modules():
    """
    Import bb_datasets' fraud modules without permanently stealing the
    top-level `datasets` package name from this repo.
    """
    preserved: dict[str, object] = {}
    bb_modules: dict[str, object] = {}

    for name in list(sys.modules):
        if name == "datasets" or name.startswith("datasets."):
            preserved[name] = sys.modules[name]
            del sys.modules[name]

    sys.path.insert(0, str(_BB_PATH))
    try:
        load_mod = importlib.import_module("fraud.load")
        features_mod = importlib.import_module("fraud.features")
        for name, module in list(sys.modules.items()):
            if name == "datasets" or name.startswith("datasets."):
                bb_modules[name] = module
    finally:
        try:
            sys.path.remove(str(_BB_PATH))
        except ValueError:
            pass
        for name in list(bb_modules):
            sys.modules.pop(name, None)
        sys.modules.update(preserved)

    return load_mod.load_transactions, features_mod.build_features


load_transactions, build_features = _import_bb_fraud_modules()


class FraudHandle(DatasetHandle):
    """
    DatasetHandle for the bb_datasets fraud dataset.

    eval_df() returns the last 20% of transactions sorted by txn_id.
    The fraud_flag column is the ground truth label.

    Feature engineering (build_features) is applied before splitting
    so both train and eval have the same engineered columns.
    """

    _LABEL_COLUMN   = "fraud_flag"
    _PRIMARY_METRIC = "f1_score"
    _SPLIT_RATIO    = 0.8          # 80% train, 20% eval — immutable

    def __init__(self, db_path: str | Path | None = None):
        """
        Args:
            db_path: path to sandbox.db. Defaults to
                     ../bb_datasets/exports/duckdb/sandbox.db
        """
        if db_path is None:
            db_path = _BB_PATH / "exports" / "duckdb" / "sandbox.db"
        self._db_path = Path(db_path)
        if not self._db_path.exists():
            raise FileNotFoundError(
                f"DuckDB not found at {self._db_path}. "
                f"Expected bb_datasets at {_BB_PATH}"
            )

    @cached_property
    def _full_df(self) -> pd.DataFrame:
        """
        Full dataset with features, sorted by txn_id for deterministic split.
        Cached — loaded once per FraudHandle instance.
        """
        raw = load_transactions(str(self._db_path))
        df  = build_features(raw)
        return df.sort_values("txn_id").reset_index(drop=True)

    @cached_property
    def _split_idx(self) -> int:
        return int(len(self._full_df) * self._SPLIT_RATIO)

    @property
    def meta(self) -> DatasetMeta:
        return DatasetMeta(
            name           = "bb_fraud_v1",
            domain         = "fraud",
            tier           = "bronze",
            version        = "1.0",
            label_column   = self._LABEL_COLUMN,
            primary_metric = self._PRIMARY_METRIC,
            row_count      = len(self._full_df),
            description    = "bb_datasets DuckDB fraud transactions, 80/20 split",
            extra          = {"db_path": str(self._db_path)},
        )

    def eval_df(self) -> pd.DataFrame:
        """Last 20% of transactions sorted by txn_id. Fixed. Immutable."""
        return self._full_df.iloc[self._split_idx:].reset_index(drop=True)

    def train_df(self) -> pd.DataFrame:
        """First 80% of transactions sorted by txn_id."""
        return self._full_df.iloc[: self._split_idx].reset_index(drop=True)

    def labels(self) -> list[bool]:
        """Ground truth for eval_df(). Same order as eval_df() rows."""
        return self.eval_df()[self._LABEL_COLUMN].astype(bool).tolist()


def get_handle(db_path: str | Path | None = None) -> FraudHandle:
    """
    Entry point for `main.py --dataset use_cases.fraud.handle`.

    Usage:
        uv run python main.py run --pattern rule_spike \
            --dataset use_cases.fraud.handle
    """
    return FraudHandle(db_path=db_path)
