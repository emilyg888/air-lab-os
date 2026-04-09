"""
Evaluator — locked scoring layer.

evaluate(result, handle, policy) → EvalMetrics

Authority chain:
  Dataset  declares → which metric defines "good" (primary_metric name)
  Evaluator computes → the metric value from flags and ground truth
  Pattern  cannot    → set or influence the primary metric

The pattern's job ends at RunResult.flags/scores/explanation.
Everything else is the evaluator's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from datasets.base import DatasetHandle
from patterns.base import RunResult


POLICY_PATH = Path(__file__).parent.parent / "scoring_policy.yaml"


# ---------------------------------------------------------------------------
# Supported primary metrics — dataset declares one of these names.
# Add new metrics here. The key is what goes in DatasetMeta.primary_metric.
# ---------------------------------------------------------------------------

def _f1(flags, labels, scores):
    if not any(flags):
        return 0.0
    return float(f1_score(labels, flags, zero_division=0))


def _precision(flags, labels, scores):
    if not any(flags):
        return 0.0
    return float(precision_score(labels, flags, zero_division=0))


def _recall(flags, labels, scores):
    return float(recall_score(labels, flags, zero_division=0))


def _accuracy(flags, labels, scores):
    return float(accuracy_score(labels, flags))


def _average_precision(flags, labels, scores):
    """Uses continuous scores, not binary flags — rewards calibrated confidence."""
    if not any(labels):
        return 0.0
    return float(average_precision_score(labels, scores))


SUPPORTED_METRICS: dict[str, Any] = {
    "f1_score":          _f1,
    "precision":         _precision,
    "recall":            _recall,
    "accuracy":          _accuracy,
    "average_precision": _average_precision,
}


def compute_primary_metric(
    metric_name: str,
    flags:       list[bool],
    labels:      list[bool],
    scores:      list[float],
) -> float:
    """
    Compute the primary metric for a run.

    Args:
        metric_name: value of DatasetMeta.primary_metric — e.g. "f1_score"
        flags:       binary predictions from RunResult.flags
        labels:      ground truth from DatasetHandle.labels()
        scores:      continuous predictions from RunResult.scores

    Returns:
        float in [0.0, 1.0]

    Raises:
        ValueError: if metric_name is not in SUPPORTED_METRICS
    """
    if metric_name not in SUPPORTED_METRICS:
        raise ValueError(
            f"Unknown primary_metric '{metric_name}'. "
            f"Valid values: {sorted(SUPPORTED_METRICS.keys())}. "
            f"Check DatasetMeta.primary_metric in your DatasetHandle "
            f"or datasets/metadata.json."
        )
    fn = SUPPORTED_METRICS[metric_name]
    y_true  = [int(l) for l in labels]
    y_pred  = [int(f) for f in flags]
    y_score = [float(s) for s in scores]
    value   = fn(y_pred, y_true, y_score)
    return max(0.0, min(1.0, float(value)))


# ---------------------------------------------------------------------------
# Scoring policy
# ---------------------------------------------------------------------------

@dataclass
class ScoringPolicy:
    primary_metric_weight: float
    explainability_weight: float
    latency_weight:        float
    cost_weight:           float
    latency_max_ms:        float
    cost_max_per_1k:       float
    working_threshold:     float
    stable_threshold:      float
    min_runs:              int

    def validate(self) -> None:
        total = (
            self.primary_metric_weight
            + self.explainability_weight
            + self.latency_weight
            + self.cost_weight
        )
        assert abs(total - 1.0) < 1e-6, (
            f"Weights must sum to 1.0, got {total:.6f}. "
            f"Check scoring_policy.yaml."
        )


def load_policy(path: Path = POLICY_PATH) -> ScoringPolicy:
    """Load and validate the scoring policy. Raises on bad weights."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    w = raw["weights"]
    policy = ScoringPolicy(
        primary_metric_weight = w["primary_metric"],
        explainability_weight = w["explainability"],
        latency_weight        = w["latency"],
        cost_weight           = w["cost"],
        latency_max_ms        = raw["latency"]["max_ms"],
        cost_max_per_1k       = raw["cost"]["max_per_1k"],
        working_threshold     = raw["promotion"]["working_threshold"],
        stable_threshold      = raw["promotion"]["stable_threshold"],
        min_runs              = raw["promotion"]["min_runs"],
    )
    policy.validate()
    return policy


# ---------------------------------------------------------------------------
# Evaluation output
# ---------------------------------------------------------------------------

@dataclass
class EvalMetrics:
    """
    Full scoring breakdown for one experiment run.

    primary_metric_name  — which metric was used (from DatasetMeta)
    primary_metric_value — computed by evaluator from flags + ground truth
    explainability_score — fraction of flagged rows with non-empty explanation
    latency_score        — 1.0 = instant, 0.0 = latency_max_ms or slower
    cost_score           — 1.0 = free, 0.0 = cost_max_per_1k or more expensive
    score                — weighted composite [0.0, 1.0] — drives promotion
    extra                — supplementary metrics from RunResult.extra_metrics
    """
    primary_metric_name:  str
    primary_metric_value: float
    explainability_score: float
    latency_score:        float
    cost_score:           float
    score:                float
    extra:                dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    result: RunResult,
    handle: DatasetHandle,
    policy: ScoringPolicy | None = None,
) -> EvalMetrics:
    """
    Score a RunResult against ground truth.

    The primary metric is determined by handle.meta.primary_metric and
    computed from result.flags and handle.labels(). The pattern has no
    input into this computation.

    Args:
        result: output of PatternHandler.detect()
        handle: DatasetHandle — provides labels() and meta.primary_metric
        policy: loaded ScoringPolicy; loads from yaml if None

    Returns:
        EvalMetrics with all dimension scores and the weighted composite

    Raises:
        AssertionError: if result.flags length != handle.labels() length
        ValueError: if handle.meta.primary_metric is not a supported metric
    """
    if policy is None:
        policy = load_policy()

    labels = handle.labels()
    assert len(result.flags) == len(labels), (
        f"RunResult has {len(result.flags)} flags "
        f"but dataset has {len(labels)} labels"
    )

    # --- Primary metric — evaluator computes, dataset declares, pattern cannot set ---
    metric_name = handle.meta.primary_metric
    primary_value = compute_primary_metric(
        metric_name = metric_name,
        flags       = result.flags,
        labels      = labels,
        scores      = result.scores,
    )

    # --- Explainability ---
    # Fraction of flagged rows that have a non-empty explanation string.
    # If nothing is flagged: 1.0 (nothing to explain — not penalised).
    flagged = [i for i, f in enumerate(result.flags) if f]
    if flagged:
        explained    = sum(1 for i in flagged if result.explanation[i].strip())
        expl_score   = explained / len(flagged)
    else:
        expl_score   = 1.0

    # --- Latency ---
    latency_score = max(0.0, 1.0 - result.latency_ms / policy.latency_max_ms)

    # --- Cost ---
    cost_score = max(0.0, 1.0 - result.cost_per_1k / policy.cost_max_per_1k)

    # --- Weighted composite ---
    composite = (
        policy.primary_metric_weight * primary_value
        + policy.explainability_weight * expl_score
        + policy.latency_weight        * latency_score
        + policy.cost_weight           * cost_score
    )

    return EvalMetrics(
        primary_metric_name  = metric_name,
        primary_metric_value = round(primary_value, 4),
        explainability_score = round(expl_score, 4),
        latency_score        = round(latency_score, 4),
        cost_score           = round(cost_score, 4),
        score                = round(composite, 4),
        extra                = result.extra_metrics or {},
    )
