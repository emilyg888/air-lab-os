"""
DatasetHandle — abstract contract for all dataset plugins.

Every dataset a use case provides must implement this interface.
The engine calls these methods and never touches raw data files.

The eval split MUST be fixed and deterministic — same rows every
call, regardless of when the dataset was generated or loaded.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class DatasetMeta:
    """
    Metadata every dataset must declare about itself.

    The engine uses this to log experiments correctly and to
    determine which scoring metric to treat as the primary metric.
    """
    name: str                        # unique dataset id, e.g. "bb_fraud_v3"
    domain: str                      # use-case domain, e.g. "fraud"
    tier: str                        # "bronze" | "silver" | "gold" | "test"
    version: str                     # e.g. "1.0", "2.0"
    label_column: str                # ground truth column in eval_df
    primary_metric: str              # metric name the evaluator treats as primary
                                     # e.g. "f1_score", "accuracy", "conversion_rate"
    row_count: int                   # total rows in the full dataset
    description: str = ""
    extra: dict[str, Any] = field(default_factory=dict)  # domain-specific metadata


class DatasetHandle(ABC):
    """
    Abstract interface for all dataset plugins.

    Implementations live in use-case repos (e.g. bb_datasets/fraud/load.py).
    The engine imports a DatasetHandle — never raw data files.

    The 80/20 train/eval split is fixed inside each implementation.
    Never expose the split ratio as a parameter — it must be immutable.
    """

    @property
    @abstractmethod
    def meta(self) -> DatasetMeta:
        """Return dataset metadata. Called once at startup for logging."""
        ...

    @abstractmethod
    def eval_df(self) -> pd.DataFrame:
        """
        Return the fixed evaluation slice (last 20% sorted by primary key).

        This slice is immutable — same rows every call. The engine scores
        all patterns against this slice so results are comparable.
        """
        ...

    @abstractmethod
    def train_df(self) -> pd.DataFrame:
        """
        Return the fixed training slice (first 80% sorted by primary key).

        Patterns that need fitting (e.g. ML models) call this internally.
        The engine does not call train_df directly.
        """
        ...

    @abstractmethod
    def labels(self) -> list[bool]:
        """
        Ground truth labels for the eval slice.

        Must be the same length as eval_df() and in the same row order.
        Derived from eval_df()[meta.label_column].astype(bool).
        """
        ...

    def summary(self) -> dict:
        """Optional: return a summary dict for dashboard display."""
        return {
            "name":           self.meta.name,
            "domain":         self.meta.domain,
            "tier":           self.meta.tier,
            "rows":           self.meta.row_count,
            "label_column":   self.meta.label_column,
            "primary_metric": self.meta.primary_metric,
        }
