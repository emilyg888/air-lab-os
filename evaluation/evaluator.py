"""
Evaluator — locked scoring layer.

evaluate(result, handle, policy) → EvalMetrics

Reads weights from scoring_policy.yaml via ScoringPolicy.
Never hardcodes metric names or weights.
The primary_metric value comes from RunResult — set by the domain
pattern, not by the evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from datasets.base import DatasetHandle
from patterns.base import RunResult


POLICY_PATH = Path(__file__).parent.parent / "scoring_policy.yaml"


@dataclass
class ScoringPolicy:
    primary_metric_weight:  float
    explainability_weight:  float
    latency_weight:         float
    cost_weight:            float
    latency_max_ms:         float
    cost_max_per_1k:        float
    working_threshold:      float
    stable_threshold:       float
    min_runs:               int

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
        explainability_weight  = w["explainability"],
        latency_weight         = w["latency"],
        cost_weight            = w["cost"],
        latency_max_ms         = raw["latency"]["max_ms"],
        cost_max_per_1k        = raw["cost"]["max_per_1k"],
        working_threshold      = raw["promotion"]["working_threshold"],
        stable_threshold       = raw["promotion"]["stable_threshold"],
        min_runs               = raw["promotion"]["min_runs"],
    )
    policy.validate()
    return policy


@dataclass
class EvalMetrics:
    """
    Full scoring breakdown for one experiment run.
    `score` is the weighted composite used for registry promotion.
    `primary_metric_value` is domain-specific (e.g. f1, accuracy).
    """
    primary_metric_value:  float    # from RunResult (domain sets this)
    primary_metric_score:  float    # normalised [0,1] — same value here
    explainability_score:  float    # fraction of flags with explanation
    latency_score:         float    # 1.0 = instant, 0.0 = max_ms
    cost_score:            float    # 1.0 = free, 0.0 = max cost
    score:                 float    # weighted composite [0, 1]
    extra: dict = None              # domain-specific metrics from RunResult


def evaluate(
    result: RunResult,
    handle: DatasetHandle,
    policy: ScoringPolicy | None = None,
) -> EvalMetrics:
    """
    Score a RunResult against ground truth labels.

    Args:
        result: output of PatternHandler.detect()
        handle: DatasetHandle providing labels() for ground truth
        policy: loaded ScoringPolicy; loads from yaml if None

    Returns:
        EvalMetrics with per-dimension scores and weighted composite
    """
    if policy is None:
        policy = load_policy()

    labels = handle.labels()
    assert len(result.flags) == len(labels), (
        f"RunResult has {len(result.flags)} flags "
        f"but dataset has {len(labels)} labels"
    )

    # --- Primary metric ---
    primary_score = float(result.primary_metric_value)
    primary_score = max(0.0, min(1.0, primary_score))

    # --- Explainability ---
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
        policy.primary_metric_weight  * primary_score
        + policy.explainability_weight * expl_score
        + policy.latency_weight        * latency_score
        + policy.cost_weight           * cost_score
    )

    return EvalMetrics(
        primary_metric_value  = round(primary_score, 4),
        primary_metric_score  = round(primary_score, 4),
        explainability_score  = round(expl_score, 4),
        latency_score         = round(latency_score, 4),
        cost_score            = round(cost_score, 4),
        score                 = round(composite, 4),
        extra                 = result.extra_metrics or {},
    )
