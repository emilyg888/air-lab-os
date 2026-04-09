"""Arena execution-mode candidate selection."""

from __future__ import annotations

from core.registry import load_registry


def select_execution_candidates() -> list[dict]:
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


def get_gold_patterns() -> list[dict]:
    return select_execution_candidates()
