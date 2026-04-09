"""ScoringPolicy loader — reads scoring_policy.yaml."""

from __future__ import annotations

from pathlib import Path

from core.evaluation import ScoringPolicy, load_policy as load_scoring_policy

POLICY_PATH = Path(__file__).parent.parent / "scoring_policy.yaml"


def load_policy(path: Path = POLICY_PATH) -> ScoringPolicy:
    """Load and validate the scoring policy. Raises on bad weights."""
    return load_scoring_policy(path)
