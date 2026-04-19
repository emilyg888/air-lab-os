"""
ErgHandle — DatasetHandle for Concept2 rowing telemetry.

Wraps the ERGBootCamp DuckDB (`daily_metrics` table) for use with the
air-lab-os engine. The engine sees only DatasetHandle — it never imports
from the ERGBootCamp repo directly.

The semantic layer exposed here is deliberately richer than the raw
`workout_sessions` schema: each row carries derived fields (delta,
rolling split, consistency, weekly load, session type) so patterns can
combine them into richer signals. Growing this semantic layer is the
engine's lever for improving detection.

Task: detect sessions that warrant concern — where the athlete entered
a fatigue or caution state. Ground truth label is derived from the
existing `fatigue_flag` heuristic but broadened to include 'caution',
giving patterns room to improve on the current hand-written rule.

Split: 50/50 by workout_date. Fixed and deterministic.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

import duckdb
import pandas as pd

from datasets.base import DatasetHandle, DatasetMeta


_DEFAULT_DB = Path(
    "/Users/emilygao/LocalDocuments/Projects/ERGBootCamp/db/rowing.duckdb"
)


class ErgHandle(DatasetHandle):
    """
    Concept2 erg dataset — one row per workout session, labelled with
    whether that session registered a caution or fatigue event.

    Columns exposed on eval_df()/train_df():
        workout_date, avg_split_sec, duration_sec, distance_m, delta,
        rolling_avg_split, consistency, weekly_load_min, session_type,
        is_fatigue (bool label)
    """

    _LABEL_COLUMN   = "is_fatigue"
    _PRIMARY_METRIC = "f1_score"
    _SPLIT_RATIO    = 0.5  # 5 train / 5 eval on current data. Immutable.

    def __init__(self, db_path: str | Path | None = None):
        self._db_path = Path(db_path) if db_path else _DEFAULT_DB
        if not self._db_path.exists():
            raise FileNotFoundError(
                f"Rowing DuckDB not found at {self._db_path}. "
                "Populate via ERGBootCamp pipelines (pull_concept2, "
                "build_daily_metrics) first."
            )

    @cached_property
    def _full_df(self) -> pd.DataFrame:
        """Load daily_metrics once per handle instance, sorted by date."""
        con = duckdb.connect(str(self._db_path), read_only=True)
        try:
            df = con.execute(
                """
                SELECT
                    workout_date,
                    avg_split_sec,
                    duration_sec,
                    distance_m,
                    delta,
                    rolling_avg_split,
                    consistency,
                    weekly_load_min,
                    session_type,
                    fatigue_flag
                FROM daily_metrics
                ORDER BY workout_date
                """
            ).df()
        finally:
            con.close()

        df["is_fatigue"] = df["fatigue_flag"].isin(["fatigue", "caution"])
        # Fill NaN derived columns with neutral values so rule patterns
        # can evaluate rows near the start of the series.
        df["delta"] = df["delta"].fillna(0.0)
        df["consistency"] = df["consistency"].fillna(0.0)
        df["weekly_load_min"] = df["weekly_load_min"].fillna(0.0)
        return df.reset_index(drop=True)

    @cached_property
    def _split_idx(self) -> int:
        return int(len(self._full_df) * self._SPLIT_RATIO)

    @property
    def meta(self) -> DatasetMeta:
        return DatasetMeta(
            name           = "concept2_erg_v1",
            domain         = "rowing",
            tier           = "bronze",
            version        = "1.0",
            label_column   = self._LABEL_COLUMN,
            primary_metric = self._PRIMARY_METRIC,
            row_count      = len(self._full_df),
            description    = (
                "Concept2 daily_metrics sessions; label = session entered "
                "fatigue or caution state."
            ),
            extra          = {"db_path": str(self._db_path)},
        )

    def eval_df(self) -> pd.DataFrame:
        return self._full_df.iloc[self._split_idx:].reset_index(drop=True)

    def train_df(self) -> pd.DataFrame:
        return self._full_df.iloc[: self._split_idx].reset_index(drop=True)

    def labels(self) -> list[bool]:
        return self.eval_df()[self._LABEL_COLUMN].astype(bool).tolist()


def get_handle(db_path: str | Path | None = None) -> ErgHandle:
    """Entry point for `main.py --dataset use_cases.concept2.handle`."""
    return ErgHandle(db_path=db_path)
