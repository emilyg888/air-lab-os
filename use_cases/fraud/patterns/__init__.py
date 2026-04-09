"""Shared utilities for fraud patterns."""

from __future__ import annotations
from sklearn.metrics import f1_score as sklearn_f1


def compute_f1(flags: list[bool], labels: list[bool]) -> float:
    """
    Compute binary F1 score. Returns 0.0 if no positives predicted or true.
    Never raises — safe to call with all-False flags.
    """
    if not any(flags) and not any(labels):
        return 1.0   # trivial: nothing to find, found nothing
    if not any(flags) or not any(labels):
        return 0.0
    return float(sklearn_f1(labels, flags, zero_division=0))
