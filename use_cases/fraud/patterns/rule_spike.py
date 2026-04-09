"""
rule_spike — flags transactions where |amount| exceeds threshold * account_mean.

Wraps the logic from bb_datasets detect_fraud_v2 (large spike rule).
Threshold is a configurable hyperparameter — tune it across runs.

Primary metric: F1 score against fraud_flag ground truth.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Append (not insert at 0) so air-lab-os `datasets/` package wins over
# bb_datasets's own `datasets/` package.
_BB_PATH = Path(__file__).parent.parent.parent.parent.parent / "bb_datasets"
if str(_BB_PATH) not in sys.path:
    sys.path.append(str(_BB_PATH))

from datasets.base import DatasetHandle
from patterns.base import PatternHandler, RunResult
from use_cases.fraud.patterns import compute_f1


class RuleSpike(PatternHandler):
    """
    Flag rows where |amount| > threshold × per-account mean |amount|.

    This is the core spike detection signal — isolates unusually large
    transactions relative to each account's normal spending baseline.

    Args:
        threshold: multiplier above account mean that triggers a flag.
                   Default 5.0. Tune upward to reduce false positives.
    """

    name    = "rule_spike"
    version = "0.1"

    def __init__(self, threshold: float = 5.0):
        self.threshold = threshold

    def run(self, handle: DatasetHandle) -> RunResult:
        df     = handle.eval_df()
        labels = handle.labels()
        n      = len(df)

        # Compute per-account mean absolute amount from the eval slice.
        # Note: using eval slice only — no train leakage for a rule-based pattern.
        abs_amt      = df["amount"].abs()
        account_mean = df.groupby("account_id")["amount"].transform(
            lambda x: x.abs().mean()
        )

        flags: list[bool] = []
        scores: list[float] = []
        explanation: list[str] = []

        for i in range(n):
            amt  = abs_amt.iloc[i]
            mean = account_mean.iloc[i]
            ratio = (amt / mean) if mean > 0 else 0.0
            flagged = ratio > self.threshold

            flags.append(bool(flagged))
            scores.append(float(min(1.0, ratio / (self.threshold * 2))) if flagged else 0.0)
            if flagged:
                explanation.append(
                    f"amount {amt:.0f} is {ratio:.1f}x account mean {mean:.0f}"
                )
            else:
                explanation.append("")

        result = RunResult(
            flags=flags, scores=scores, explanation=explanation
        )
        result.primary_metric_value = compute_f1(flags, labels)
        result.extra_metrics = {
            "precision": float(_precision(flags, labels)),
            "recall":    float(_recall(flags, labels)),
            "threshold": self.threshold,
            "n_flagged": int(sum(flags)),
        }
        return result

    def describe(self) -> dict:
        return {"pattern": self.name, "version": self.version, "threshold": self.threshold}


def _precision(flags: list[bool], labels: list[bool]) -> float:
    tp = sum(f and l for f, l in zip(flags, labels))
    fp = sum(f and not l for f, l in zip(flags, labels))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def _recall(flags: list[bool], labels: list[bool]) -> float:
    tp = sum(f and l for f, l in zip(flags, labels))
    fn = sum(not f and l for f, l in zip(flags, labels))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def get_pattern() -> RuleSpike:
    """Entry point for engine pattern discovery."""
    return RuleSpike()
