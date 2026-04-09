"""ScoringPolicy loader — reads scoring_policy.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml

from evaluation.evaluator import ScoringPolicy

POLICY_PATH = Path(__file__).parent.parent / "scoring_policy.yaml"


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
