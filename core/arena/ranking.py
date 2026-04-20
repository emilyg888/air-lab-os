"""Arena leaderboard and ranking views."""

from __future__ import annotations

from core.registry import load_registry
from runtime.config import load_policy


def compute_rank_score(
    entry: dict,
    *,
    stable_bonus: float,
    unstable_bonus: float,
) -> float:
    avg_score = entry.get("avg_score", 0.0)
    confidence = entry.get("confidence", 0.0)
    is_stable = entry.get("is_stable", False)

    stability_bonus = stable_bonus if is_stable else unstable_bonus
    return avg_score * confidence * stability_bonus


def rank_patterns() -> list[dict]:
    policy = load_policy()
    registry = load_registry()
    ranked: list[dict] = []

    for name, entry in registry.items():
        ranked.append(
            {
                "pattern": name,
                "rank_score": round(
                    compute_rank_score(
                        entry,
                        stable_bonus=policy.ranking_stable_bonus,
                        unstable_bonus=policy.ranking_unstable_bonus,
                    ),
                    4,
                ),
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
