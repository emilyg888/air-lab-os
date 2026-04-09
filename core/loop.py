# core/loop.py

from __future__ import annotations

from typing import List, Dict, Any, Mapping

from core.evaluation import evaluate


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


def _working_threshold(policy: Mapping[str, Any] | Any) -> float:
    if isinstance(policy, Mapping):
        return policy["promotion"]["working_threshold"]
    return float(policy.working_threshold)


def run_lab(
    patterns: List[Any],
    explore_dataset: str,
    validate_dataset: str,
    policy: Dict,
    mode: str = "exploration",  # "exploration" | "execution"
) -> List[Dict]:
    """
    Core AI Lab loop.

    Exploration Mode:
        - Runs on explore dataset (e.g. fraud_v1, synthetic)
        - Loose filtering

    Execution Mode:
        - Runs on validation dataset (e.g. fraud_gold)
        - Strict enforcement

    Returns:
        List of results for reporting / inspection
    """

    explore_ds = _load_dataset(explore_dataset)
    validate_ds = _load_dataset(validate_dataset)

    results = []

    for pattern in patterns:
        pattern_name = getattr(pattern, "name", pattern.__class__.__name__)

        # -----------------------------
        # Phase 1 — Exploration
        # -----------------------------
        result_explore = _run_experiment(pattern, explore_ds)
        eval_explore = evaluate(result_explore, explore_ds, policy)

        # Early pruning (only in exploration mode)
        if mode == "exploration":
            if eval_explore.score < _working_threshold(policy):
                continue

        # -----------------------------
        # Phase 2 — Validation (Gold)
        # -----------------------------
        result_validate = _run_experiment(pattern, validate_ds)
        eval_validate = evaluate(result_validate, validate_ds, policy)

        # -----------------------------
        # Stability check (optional)
        # -----------------------------
        stability_gap = abs(eval_explore.score - eval_validate.score)

        is_stable = stability_gap < 0.2

        # -----------------------------
        # Registry update (ONLY gold matters)
        # -----------------------------
        _update_registry(
            pattern_name=pattern_name,
            score=eval_validate.score,
            metadata={
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
