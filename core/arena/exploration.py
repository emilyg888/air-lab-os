"""Arena exploration candidate selection."""

from __future__ import annotations

from core.registry import load_registry
from runtime.config import load_policy


def select_exploration_candidates(
    top_n: int = 3,
    max_runs_per_pattern: int | None = None,
) -> list[dict]:
    policy = load_policy()
    if max_runs_per_pattern is None:
        max_runs_per_pattern = policy.exploration_max_runs_per_pattern
    registry = load_registry()
    candidates: list[dict] = []

    for name, entry in registry.items():
        if entry.get("status") == "gold":
            continue

        runs = entry.get("runs", 0)
        if runs >= max_runs_per_pattern:
            continue

        avg_score = entry.get("avg_score", 0.0)
        exploration_score = (1 / (runs + 1)) * (1 + avg_score)

        candidates.append(
            {
                "pattern": name,
                "exploration_score": round(exploration_score, 4),
                "runs": runs,
                "status": entry.get("status"),
                "avg_score": avg_score,
            }
        )

    candidates.sort(key=lambda item: item["exploration_score"], reverse=True)
    return candidates[:top_n]
