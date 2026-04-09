"""
rule_velocity — flags transactions that are part of a same-timestamp burst.

Wraps the burst detection logic from bb_datasets (velocity burst pattern):
flags DEBIT transactions when the same account has >= burst_count DEBITs
at the exact same timestamp.

Primary metric: F1 score against fraud_flag ground truth.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from datasets.base import DatasetHandle
from patterns.base import PatternHandler, RunResult


class RuleVelocity(PatternHandler):
    """
    Flag DEBIT transactions in same-timestamp bursts per account.

    A burst is defined as >= burst_count DEBIT transactions from the same
    account_id at the exact same timestamp. All transactions in the burst
    are flagged.

    Args:
        burst_count: minimum number of same-ts DEBITs to trigger.
                     Default 3. Lower = more sensitive, higher = fewer FP.
    """

    name    = "rule_velocity"
    version = "0.1"

    def __init__(self, burst_count: int = 3):
        self.burst_count = burst_count

    def run(self, handle: DatasetHandle) -> RunResult:
        df     = handle.eval_df()
        labels = handle.labels()
        n      = len(df)

        # Count DEBITs per (account_id, timestamp) group
        debit_mask = df["txn_type"] == "DEBIT"
        same_ts_counts = (
            df[debit_mask]
            .groupby(["account_id", "timestamp"])["txn_id"]
            .transform("count")
        )
        # Reindex to full df (non-DEBIT rows get NaN → fill 0)
        same_ts_counts = same_ts_counts.reindex(df.index, fill_value=0)

        flags: list[bool] = []
        scores: list[float] = []
        explanation: list[str] = []

        for i in range(n):
            count   = int(same_ts_counts.iloc[i])
            is_debit = debit_mask.iloc[i]
            flagged  = is_debit and count >= self.burst_count

            flags.append(bool(flagged))
            scores.append(float(min(1.0, count / (self.burst_count * 2))) if flagged else 0.0)
            if flagged:
                explanation.append(
                    f"burst: {count} DEBITs at same timestamp for account "
                    f"{df['account_id'].iloc[i]}"
                )
            else:
                explanation.append("")

        result = RunResult(
            flags=flags, scores=scores, explanation=explanation
        )
        # Do NOT set primary_metric_value — evaluator owns primary metric
        result.extra_metrics = {
            "precision":   float(_precision(flags, labels)),
            "recall":      float(_recall(flags, labels)),
            "burst_count": self.burst_count,
            "n_flagged":   int(sum(flags)),
        }
        return result

    def describe(self) -> dict:
        return {"pattern": self.name, "version": self.version, "burst_count": self.burst_count}


def _precision(flags: list[bool], labels: list[bool]) -> float:
    tp = sum(f and l for f, l in zip(flags, labels))
    fp = sum(f and not l for f, l in zip(flags, labels))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def _recall(flags: list[bool], labels: list[bool]) -> float:
    tp = sum(f and l for f, l in zip(flags, labels))
    fn = sum(not f and l for f, l in zip(flags, labels))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def get_pattern() -> RuleVelocity:
    """Entry point for engine pattern discovery."""
    return RuleVelocity()
