"""
Evaluation engine — locked scoring layer.

evaluate(result, labels) → float in [0.0, 1.0]

Reads weights from scoring_policy.yaml. Do not hardcode weights here.
The policy file is the single source of truth.
"""

from __future__ import annotations
import yaml
from pathlib import Path
from dataclasses import dataclass
from sklearn.metrics import precision_score, recall_score, f1_score
from src.detectors.base import DetectionResult


POLICY_PATH = Path(__file__).parent.parent.parent / "scoring_policy.yaml"


@dataclass
class ScoringPolicy:
    precision_recall_weight: float
    explainability_weight: float
    latency_weight: float
    cost_weight: float
    latency_max_ms: float
    cost_max_per_1k: float
    silver_threshold: float
    gold_threshold: float
    min_runs: int

    def validate(self) -> None:
        total = (
            self.precision_recall_weight
            + self.explainability_weight
            + self.latency_weight
            + self.cost_weight
        )
        assert abs(total - 1.0) < 1e-6, (
            f"Scoring weights must sum to 1.0, got {total:.4f}. "
            f"Check scoring_policy.yaml."
        )


def load_policy(path: Path = POLICY_PATH) -> ScoringPolicy:
    """Load and validate the scoring policy. Raises on missing file or bad weights."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    w = raw["weights"]
    policy = ScoringPolicy(
        precision_recall_weight=w["precision_recall"],
        explainability_weight=w["explainability"],
        latency_weight=w["latency"],
        cost_weight=w["cost"],
        latency_max_ms=raw["latency"]["max_ms"],
        cost_max_per_1k=raw["cost"]["max_per_1k"],
        silver_threshold=raw["promotion"]["silver_threshold"],
        gold_threshold=raw["promotion"]["gold_threshold"],
        min_runs=raw["promotion"]["min_runs"],
    )
    policy.validate()
    return policy


@dataclass
class EvaluationMetrics:
    """Full breakdown of the evaluation. Score is the weighted composite."""
    precision: float
    recall: float
    f1: float
    precision_recall_score: float   # normalised [0,1], same as f1 here
    explainability_score: float     # fraction of flags with non-empty explanation
    latency_score: float            # 1.0 = 0ms, 0.0 = max_ms
    cost_score: float               # 1.0 = free, 0.0 = max cost
    score: float                    # weighted composite


def evaluate(
    result: DetectionResult,
    labels: list[bool],
    policy: ScoringPolicy | None = None,
) -> EvaluationMetrics:
    """
    Score a DetectionResult against ground truth labels.

    Args:
        result: output of Detector.detect()
        labels: ground truth fraud flags, same length as result.flags
        policy: loaded ScoringPolicy; loads from yaml if None

    Returns:
        EvaluationMetrics with per-dimension scores and weighted composite
    """
    if policy is None:
        policy = load_policy()

    assert len(result.flags) == len(labels), (
        f"result has {len(result.flags)} flags but labels has {len(labels)}"
    )

    # --- Precision / recall ---
    y_true = [int(l) for l in labels]
    y_pred = [int(f) for f in result.flags]

    if sum(y_pred) == 0:
        # Nothing flagged — precision undefined, recall = 0
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

    pr_score = float(f1)  # use F1 as the precision/recall composite

    # --- Explainability ---
    flagged_indices = [i for i, f in enumerate(result.flags) if f]
    if flagged_indices:
        with_explanation = sum(
            1 for i in flagged_indices if result.explanation[i].strip()
        )
        expl_score = with_explanation / len(flagged_indices)
    else:
        expl_score = 1.0  # nothing flagged = nothing to explain

    # --- Latency ---
    latency_score = max(
        0.0, 1.0 - result.latency_ms / policy.latency_max_ms
    )

    # --- Cost ---
    cost_score = max(
        0.0, 1.0 - result.cost_per_1k / policy.cost_max_per_1k
    )

    # --- Weighted composite ---
    composite = (
        policy.precision_recall_weight * pr_score
        + policy.explainability_weight * expl_score
        + policy.latency_weight * latency_score
        + policy.cost_weight * cost_score
    )

    return EvaluationMetrics(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        precision_recall_score=round(pr_score, 4),
        explainability_score=round(expl_score, 4),
        latency_score=round(latency_score, 4),
        cost_score=round(cost_score, 4),
        score=round(composite, 4),
    )
