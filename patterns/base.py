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
    Standardised output of every pattern run.

    All three lists must be the same length as the eval DataFrame.

    flags       — True if the row is a positive detection
    scores      — float in [0.0, 1.0], confidence or probability
    explanation — human-readable reason string, "" if none
    latency_ms  — wall-clock ms for the full detect() call (set by base)
    cost_per_1k — estimated USD per 1000 rows (0.0 for local patterns)
    primary_metric_value — the domain metric for this run (e.g. f1 score).
                           Set by the engine after scoring, not by the pattern.
    """
    flags: list[bool]
    scores: list[float]
    explanation: list[str]
    latency_ms: float = 0.0
    cost_per_1k: float = 0.0
    primary_metric_value: float = 0.0
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

    Pattern files implementing this class live in:
      patterns/scratch/   — exploration, unproven
      patterns/working/   — improving, above working_threshold
      patterns/stable/    — promoted, certified on gold dataset

    Usage:
        handle  = MyDatasetHandle()
        pattern = MyPattern(threshold=0.5)
        result  = pattern.detect(handle)
    """

    name: str = "base"                   # unique pattern identifier
    version: str = "0.1"                 # pattern version string

    def detect(self, handle: DatasetHandle) -> RunResult:
        """
        Public entry point. Times the call, injects latency_ms.
        Do NOT override — override run() instead.
        """
        t0 = time.perf_counter()
        result = self.run(handle)
        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    @abstractmethod
    def run(self, handle: DatasetHandle) -> RunResult:
        """
        Implement pattern logic here.

        Args:
            handle: DatasetHandle providing eval_df(), train_df(), labels().
                    The pattern must not assume any specific columns exist
                    beyond what DatasetHandle.meta declares.

        Returns:
            RunResult with flags, scores, explanation the same length as
            handle.eval_df(). Do NOT set latency_ms — base class does it.
        """
        ...

    def describe(self) -> dict:
        """
        Return hyperparameter dict for run logging.
        Override to include pattern-specific config.
        """
        return {"pattern": self.name, "version": self.version}
