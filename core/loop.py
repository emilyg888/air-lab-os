# core/loop.py

from __future__ import annotations

from typing import List, Dict, Any

from core.evaluation import evaluate
from core.mode import Mode, normalize_mode


def _load_dataset(dataset_name: str) -> Any:
    from core.dataset_loader import load_dataset

    return load_dataset(dataset_name)


def _run_experiment(pattern: Any, dataset: Any) -> Any:
    from core.experiment import run_experiment

    return run_experiment(pattern, dataset)


def _update_registry(
    pattern_name: str,
    score: float,
    metadata: dict[str, Any],
    policy: dict[str, Any],
) -> None:
    from core.registry import update_registry

    update_registry(
        pattern_name=pattern_name,
        score=score,
        metadata=metadata,
        policy=policy,
    )


def _stability_gap_threshold(policy: dict[str, Any] | Any) -> float:
    if isinstance(policy, dict):
        return float(policy["rules"]["stability_gap_threshold"])
    return float(policy.stability_gap_threshold)


def run_lab(
    patterns: List[Any],
    explore_dataset: str,
    validate_dataset: str,
    policy: Dict,
    mode: Mode | str = Mode.EXPLORATION,
) -> List[Dict]:
    """
    Core AI Lab execution loop.

    The loop executes the supplied patterns against the supplied datasets.
    It does not rank, prune, or choose which patterns to run.

    Returns:
        List of results for reporting / inspection
    """

    normalized_mode = normalize_mode(mode)
    explore_ds = _load_dataset(explore_dataset)
    validate_ds = _load_dataset(validate_dataset)

    results = []

    for pattern in patterns:
        pattern_name = getattr(pattern, "name", pattern.__class__.__name__)

        # -----------------------------
        # Phase 1 — Exploration
        # -----------------------------
        result_explore = _run_experiment(pattern, explore_ds)
        eval_explore = evaluate(result_explore, explore_ds, policy, mode=normalized_mode)

        # -----------------------------
        # Phase 2 — Validation (Gold)
        # -----------------------------
        result_validate = _run_experiment(pattern, validate_ds)
        eval_validate = evaluate(result_validate, validate_ds, policy, mode=normalized_mode)

        # -----------------------------
        # Stability check (optional)
        # -----------------------------
        stability_gap = abs(eval_explore.score - eval_validate.score)

        is_stable = stability_gap < _stability_gap_threshold(policy)

        # -----------------------------
        # Registry update (ONLY gold matters)
        # -----------------------------
        _update_registry(
            pattern_name=pattern_name,
            score=eval_validate.score,
            metadata={
                "mode": normalized_mode.value,
                "explore_score": eval_explore.score,
                "stability_gap": stability_gap,
                "is_stable": is_stable,
                "dataset": validate_dataset,
            },
            policy=policy,
        )

        # -----------------------------
        # Collect result
        # -----------------------------
        results.append(
            {
                "pattern": pattern_name,
                "explore_score": eval_explore.score,
                "validation_score": eval_validate.score,
                "stability_gap": stability_gap,
                "is_stable": is_stable,
            }
        )

    return results
