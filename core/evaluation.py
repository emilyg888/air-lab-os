"""Shared evaluation entry points for loop and playground flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from datasets.base import DatasetHandle
from core.mode import Mode, normalize_mode
from patterns.base import RunResult


POLICY_PATH = Path(__file__).parent.parent / "scoring_policy.yaml"


@dataclass
class EvalResult:
    score: float
    metrics: dict[str, float]
    passed_constraints: bool


@dataclass
class ScoringPolicy:
    primary_metric_weight: float
    explainability_weight: float
    latency_weight: float
    cost_weight: float
    latency_max_ms: float
    cost_max_per_1k: float
    working_threshold: float
    stable_threshold: float
    min_runs: int
    primary_metric_floor: float
    stability_threshold: float
    penalty_factor: float
    stability_gap_threshold: float
    promotion_confidence_threshold: float
    ranking_stable_bonus: float
    ranking_unstable_bonus: float
    exploration_max_runs_per_pattern: int

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


@dataclass
class EvalMetrics:
    primary_metric_name: str
    primary_metric_value: float
    explainability_score: float
    latency_score: float
    cost_score: float
    score: float
    extra: dict[str, Any] = field(default_factory=dict)


def _f1(flags: list[bool], labels: list[bool], scores: list[float] | None) -> float:
    if not any(flags):
        return 0.0
    return float(f1_score(labels, flags, zero_division=0))


def _precision(
    flags: list[bool], labels: list[bool], scores: list[float] | None
) -> float:
    if not any(flags):
        return 0.0
    return float(precision_score(labels, flags, zero_division=0))


def _recall(
    flags: list[bool], labels: list[bool], scores: list[float] | None
) -> float:
    return float(recall_score(labels, flags, zero_division=0))


def _accuracy(
    flags: list[bool], labels: list[bool], scores: list[float] | None
) -> float:
    return float(accuracy_score(labels, flags))


def _average_precision(
    flags: list[bool], labels: list[bool], scores: list[float] | None
) -> float:
    if not any(labels):
        return 0.0
    if scores is None:
        raise ValueError("average_precision requires continuous scores")
    return float(average_precision_score(labels, scores))


SUPPORTED_METRICS: dict[str, Any] = {
    "f1_score": _f1,
    "precision": _precision,
    "recall": _recall,
    "accuracy": _accuracy,
    "average_precision": _average_precision,
}


def load_policy(path: Path = POLICY_PATH) -> ScoringPolicy:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _policy_from_mapping(raw)


def _coerce_scoring_policy(
    policy: ScoringPolicy | Mapping[str, Any] | None,
) -> ScoringPolicy:
    if policy is None:
        return load_policy()
    if isinstance(policy, ScoringPolicy):
        return policy
    return _policy_from_mapping(policy)


def _weight_value(weights: Mapping[str, Any], key: str) -> float:
    value = weights[key]
    if isinstance(value, Mapping):
        return float(value["weight"])
    return float(value)


def _constraint_value(raw: Mapping[str, Any], group: str, field: str) -> float:
    constraints = raw.get("constraints", {})
    if isinstance(constraints, Mapping):
        payload = constraints.get(group)
        if isinstance(payload, Mapping) and field in payload:
            return float(payload[field])

    legacy_group = raw.get(group)
    if isinstance(legacy_group, Mapping) and field in legacy_group:
        return float(legacy_group[field])

    raise KeyError(f"Missing policy constraint {group}.{field}")


def _rule_value(raw: Mapping[str, Any], *names: str) -> float:
    rules = raw.get("rules", {})
    if isinstance(rules, Mapping):
        for name in names:
            if name in rules:
                return float(rules[name])
    raise KeyError(f"Missing policy rule {' or '.join(names)}")


def _arena_value(raw: Mapping[str, Any], section: str, field: str) -> float:
    arena = raw.get("arena", {})
    if isinstance(arena, Mapping):
        payload = arena.get(section)
        if isinstance(payload, Mapping) and field in payload:
            return float(payload[field])
    raise KeyError(f"Missing arena policy {section}.{field}")


def _policy_from_mapping(raw: Mapping[str, Any]) -> ScoringPolicy:
    weights = raw["weights"]
    policy = ScoringPolicy(
        primary_metric_weight=_weight_value(weights, "primary_metric"),
        explainability_weight=_weight_value(weights, "explainability"),
        latency_weight=_weight_value(weights, "latency"),
        cost_weight=_weight_value(weights, "cost"),
        latency_max_ms=_constraint_value(raw, "latency", "max_ms"),
        cost_max_per_1k=_constraint_value(raw, "cost", "max_per_1k"),
        primary_metric_floor=_rule_value(raw, "primary_metric_floor"),
        stability_threshold=_rule_value(raw, "stability_threshold", "stability_variance_threshold"),
        penalty_factor=_rule_value(raw, "penalty_factor"),
        stability_gap_threshold=_rule_value(raw, "stability_gap_threshold"),
        promotion_confidence_threshold=_rule_value(raw, "promotion_confidence_threshold"),
        ranking_stable_bonus=_arena_value(raw, "ranking", "stable_bonus"),
        ranking_unstable_bonus=_arena_value(raw, "ranking", "unstable_bonus"),
        exploration_max_runs_per_pattern=int(
            _arena_value(raw, "exploration", "max_runs_per_pattern")
        ),
        working_threshold=raw["promotion"]["working_threshold"],
        stable_threshold=raw["promotion"]["stable_threshold"],
        min_runs=raw["promotion"]["min_runs"],
    )
    policy.validate()
    return policy


def compute_primary_metric(
    metric_name: str,
    flags: list[bool],
    labels: list[bool],
    scores: list[float] | None = None,
) -> float:
    if metric_name not in SUPPORTED_METRICS:
        raise ValueError(
            f"Unknown primary_metric '{metric_name}'. "
            f"Valid values: {sorted(SUPPORTED_METRICS.keys())}. "
            f"Check DatasetMeta.primary_metric in your DatasetHandle "
            f"or datasets/metadata.json."
        )

    y_true = [bool(label) for label in labels]
    y_pred = [bool(flag) for flag in flags]
    y_score = None if scores is None else [float(score) for score in scores]

    value = SUPPORTED_METRICS[metric_name](y_pred, y_true, y_score)
    return max(0.0, min(1.0, float(value)))


def check_constraints(result: Mapping[str, Any], policy: Mapping[str, Any]) -> bool:
    constraints = policy.get("constraints", {})

    latency_ok = True
    cost_ok = True

    if "latency" in constraints:
        max_latency = constraints["latency"].get("max_ms")
        latency = result.get("latency_ms", 0)
        latency_ok = latency <= max_latency

    if "cost" in constraints:
        max_cost = constraints["cost"].get("max_per_1k")
        cost = result.get("cost_per_1k", 0)
        cost_ok = cost <= max_cost

    return latency_ok and cost_ok


def _constraint_penalty(mode: Mode, penalty_factor: float) -> float:
    if mode is Mode.EXPLORATION:
        return 1.0
    return penalty_factor


def _apply_primary_metric_floor(score: float, primary_metric: float, floor: float) -> float:
    if primary_metric < floor:
        return 0.0
    return score


def _evaluate_handle_result(
    result: RunResult,
    handle: DatasetHandle,
    policy: ScoringPolicy | Mapping[str, Any] | None = None,
    mode: Mode = Mode.EXECUTION,
) -> EvalMetrics:
    policy = _coerce_scoring_policy(policy)
    mode = normalize_mode(mode)

    labels = handle.labels()
    assert len(result.flags) == len(labels), (
        f"RunResult has {len(result.flags)} flags "
        f"but dataset has {len(labels)} labels"
    )

    metric_name = handle.meta.primary_metric
    primary_value = compute_primary_metric(
        metric_name=metric_name,
        flags=result.flags,
        labels=labels,
        scores=result.scores,
    )

    flagged = [i for i, flag in enumerate(result.flags) if flag]
    if flagged:
        explained = sum(1 for i in flagged if result.explanation[i].strip())
        explainability_score = explained / len(flagged)
    else:
        explainability_score = 1.0

    latency_score = max(0.0, 1.0 - result.latency_ms / policy.latency_max_ms)
    cost_score = max(0.0, 1.0 - result.cost_per_1k / policy.cost_max_per_1k)

    composite = (
        policy.primary_metric_weight * primary_value
        + policy.explainability_weight * explainability_score
        + policy.latency_weight * latency_score
        + policy.cost_weight * cost_score
    )
    composite = _apply_primary_metric_floor(
        composite,
        primary_value,
        policy.primary_metric_floor,
    )
    constraints_payload = {
        "latency_ms": result.latency_ms,
        "cost_per_1k": result.cost_per_1k,
    }
    constraints_policy = {
        "constraints": {
            "latency": {"max_ms": policy.latency_max_ms},
            "cost": {"max_per_1k": policy.cost_max_per_1k},
        }
    }
    passed_constraints = check_constraints(constraints_payload, constraints_policy)
    if not passed_constraints:
        composite *= _constraint_penalty(mode, policy.penalty_factor)

    return EvalMetrics(
        primary_metric_name=metric_name,
        primary_metric_value=round(primary_value, 4),
        explainability_score=round(explainability_score, 4),
        latency_score=round(latency_score, 4),
        cost_score=round(cost_score, 4),
        score=round(composite, 4),
        extra=result.extra_metrics or {},
    )


def _evaluate_mapping_result(
    result: Mapping[str, Any],
    dataset: Mapping[str, Any],
    policy: Mapping[str, Any],
    mode: Mode = Mode.EXECUTION,
) -> EvalResult:
    mode = normalize_mode(mode)
    flags = result["flags"]
    labels = dataset["labels"]
    metadata = dataset.get("metadata", {})

    metric_name = metadata.get("primary_metric", "f1_score")
    primary_metric = compute_primary_metric(
        metric_name=metric_name,
        flags=flags,
        labels=labels,
        scores=result.get("scores"),
    )

    explainability = result.get("explainability_score", 0.5)
    latency = result.get("latency_ms", 0)
    cost = result.get("cost_per_1k", 0)

    constraints = policy.get("constraints", {})
    latency_score = (
        max(0.0, 1 - (latency / constraints["latency"]["max_ms"]))
        if "latency" in constraints
        else 1.0
    )
    cost_score = (
        max(0.0, 1 - (cost / constraints["cost"]["max_per_1k"]))
        if "cost" in constraints
        else 1.0
    )

    weights = policy["weights"]
    final_score = (
        weights["primary_metric"]["weight"] * primary_metric
        + weights["explainability"]["weight"] * explainability
        + weights["latency"]["weight"] * latency_score
        + weights["cost"]["weight"] * cost_score
    )
    floor = _rule_value(policy, "primary_metric_floor")
    final_score = _apply_primary_metric_floor(final_score, primary_metric, floor)

    passed_constraints = check_constraints(result, policy)
    if not passed_constraints:
        final_score *= _constraint_penalty(mode, _rule_value(policy, "penalty_factor"))

    return EvalResult(
        score=round(final_score, 4),
        metrics={
            "primary_metric": primary_metric,
            "explainability": explainability,
            "latency_score": latency_score,
            "cost_score": cost_score,
        },
        passed_constraints=passed_constraints,
    )


def evaluate(
    result: RunResult | Mapping[str, Any],
    target: DatasetHandle | Mapping[str, Any],
    policy: ScoringPolicy | Mapping[str, Any] | None = None,
    mode: Mode | str = Mode.EXECUTION,
) -> EvalMetrics | EvalResult:
    normalized_mode = normalize_mode(mode)
    if isinstance(target, DatasetHandle):
        return _evaluate_handle_result(result, target, policy, mode=normalized_mode)

    if policy is None:
        raise ValueError("policy is required for dict-based evaluation")
    return _evaluate_mapping_result(result, target, policy, mode=normalized_mode)
