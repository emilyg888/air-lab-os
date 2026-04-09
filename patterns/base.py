"""
PatternHandler — abstract contract for all pattern plugins.

Every pattern the engine runs must implement this interface.
The engine calls detect() and never imports domain-specific code.

Pattern files live in patterns/scratch/, patterns/working/, or
patterns/stable/ depending on their current promotion tier.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from datasets.base import DatasetHandle


@dataclass
class RunResult:
    """
    Standardised output contract for every pattern run.

    All three lists must be the same length as eval_df().

    flags       — True if the row is a positive detection.
    scores      — float in [0.0, 1.0], per-row confidence or probability.
                  Higher = more likely to be a true positive.
    explanation — human-readable reason string per row. Empty string ""
                  if the row is not flagged or no explanation is available.
    latency_ms  — wall-clock milliseconds for the full detect() call.
                  Set by PatternHandler.detect(). Do NOT set in run().
    cost_per_1k — estimated USD cost per 1000 rows. 0.0 for local models.
                  Set this if the pattern calls an external API.
    extra_metrics — optional dict of supplementary diagnostic values for
                  logging (e.g. {"precision": 0.8, "n_flagged": 12}).
                  These are written to runs.json but never affect scoring.

    What patterns must NOT do:
      - Do not compute or set any performance metric (F1, accuracy, etc).
        The evaluator computes all metrics from flags and ground truth.
        The dataset declares which metric to use via DatasetMeta.primary_metric.
      - Do not set latency_ms. The base class sets it automatically.
    """
    flags:        list[bool]
    scores:       list[float]
    explanation:  list[str]
    latency_ms:   float = 0.0
    cost_per_1k:  float = 0.0
    extra_metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.flags)
        assert len(self.scores) == n, \
            f"scores length {len(self.scores)} must match flags length {n}"
        assert len(self.explanation) == n, \
            f"explanation length {len(self.explanation)} must match flags length {n}"
        assert all(0.0 <= s <= 1.0 for s in self.scores), \
            "all scores must be in [0.0, 1.0]"


class PatternHandler(ABC):
    """
    Abstract base for all pattern implementations.

    Subclasses implement run(). The base class wraps run() to record
    wall-clock latency automatically via detect().

    Pattern contract — a pattern is responsible for:
      1. Reading handle.eval_df() to get the rows to score.
      2. Producing flags, scores, explanation of the same length.
      3. Optionally reading handle.train_df() for ML patterns that fit.
      4. Optionally populating extra_metrics for diagnostic logging.

    A pattern is NOT responsible for:
      - Computing F1, accuracy, or any performance metric.
      - Knowing which metric the dataset uses.
      - Setting primary_metric_value (field removed — evaluator owns this).

    Usage:
        handle  = MyDatasetHandle()
        pattern = MyPattern(threshold=0.5)
        result  = pattern.detect(handle)   # latency_ms set automatically
    """

    name:    str = "base"    # unique pattern identifier — override in subclass
    version: str = "0.1"    # pattern version string — override in subclass

    def detect(self, handle: DatasetHandle) -> RunResult:
        """
        Public entry point. Times the call and injects latency_ms.
        Do NOT override — override run() instead.
        """
        t0 = time.perf_counter()
        result = self.run(handle)
        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    @abstractmethod
    def run(self, handle: DatasetHandle) -> RunResult:
        """
        Implement detection logic here.

        Args:
            handle: DatasetHandle providing eval_df(), train_df(), labels().
                    Do not assume specific column names beyond what
                    DatasetMeta declares.

        Returns:
            RunResult with flags, scores, explanation — same length as
            handle.eval_df(). Do NOT set latency_ms. Do NOT compute
            or set any performance metric.
        """
        ...

    def describe(self) -> dict:
        """
        Return hyperparameter dict for run logging.
        Override in subclasses to include pattern-specific config.
        """
        return {"pattern": self.name, "version": self.version}
