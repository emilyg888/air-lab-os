"""Arena ranking and candidate selection for pattern exploration."""

from __future__ import annotations

from core.registry import load_registry

DEFAULT_MAX_RUNS_PER_PATTERN = 5


def compute_rank_score(entry: dict) -> float:
    """
    Rank mature patterns for reporting and leaderboards.

    Higher avg score and confidence help. Stable patterns get a small bonus.
    """

    avg_score = entry.get("avg_score", 0.0)
    confidence = entry.get("confidence", 0.0)
    is_stable = entry.get("is_stable", False)

    stability_bonus = 1.1 if is_stable else 0.9
    return avg_score * confidence * stability_bonus


def rank_patterns() -> list[dict]:
    registry = load_registry()
    ranked: list[dict] = []

    for name, entry in registry.items():
        ranked.append(
            {
                "pattern": name,
                "rank_score": round(compute_rank_score(entry), 4),
                "avg_score": entry.get("avg_score"),
                "confidence": entry.get("confidence"),
                "status": entry.get("status"),
                "is_stable": entry.get("is_stable"),
                "runs": entry.get("runs", 0),
            }
        )

    ranked.sort(key=lambda item: item["rank_score"], reverse=True)
    return ranked


def get_top_patterns(n: int = 3) -> list[dict]:
    return rank_patterns()[:n]


def get_gold_patterns() -> list[dict]:
    registry = load_registry()
    gold = [
        {
            "pattern": name,
            "avg_score": entry.get("avg_score"),
            "confidence": entry.get("confidence"),
            "runs": entry.get("runs", 0),
            "status": entry.get("status"),
        }
        for name, entry in registry.items()
        if entry.get("status") == "gold"
    ]
    return sorted(gold, key=lambda item: item["avg_score"], reverse=True)


def select_exploration_candidates(
    top_n: int = 3,
    max_runs_per_pattern: int = DEFAULT_MAX_RUNS_PER_PATTERN,
) -> list[dict]:
    """
    Pick underexplored but promising non-gold patterns for the loop.

    Exploration score:
        (1 / (runs + 1)) * (1 + avg_score)
    """

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
