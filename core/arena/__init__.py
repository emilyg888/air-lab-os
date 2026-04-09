"""Arena decision modules."""

from core.arena.execution import get_gold_patterns, select_execution_candidates
from core.arena.exploration import select_exploration_candidates
from core.arena.ranking import compute_rank_score, get_top_patterns, rank_patterns
from core.mode import Mode, normalize_mode


def select_candidates(mode: Mode | str, *, top_n: int = 3) -> list[dict]:
    normalized_mode = normalize_mode(mode)
    if normalized_mode is Mode.EXPLORATION:
        return select_exploration_candidates(top_n=top_n)
    return select_execution_candidates()[:top_n]

__all__ = [
    "Mode",
    "compute_rank_score",
    "get_gold_patterns",
    "get_top_patterns",
    "normalize_mode",
    "rank_patterns",
    "select_candidates",
    "select_execution_candidates",
    "select_exploration_candidates",
]
