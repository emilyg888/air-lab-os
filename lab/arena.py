"""
Arena — compare_patterns() runs multiple patterns on the same dataset.

compare_patterns(patterns, handle) → ArenaResult

Returns a ranked list of patterns by composite score.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from datasets.base import DatasetHandle
from lab.playground import ExperimentResult, run_experiment
from patterns.base import PatternHandler

RUNS_PATH     = Path("memory/runs.json")
REGISTRY_PATH = Path("registry.json")


@dataclass
class ArenaResult:
    """Ranked output of a multi-pattern comparison run."""
    rankings:   list[ExperimentResult]   # sorted best → worst by score
    winner:     ExperimentResult | None  # highest scoring non-crash result
    dataset_id: str


def compare_patterns(
    patterns:      list[PatternHandler],
    handle:        DatasetHandle,
    description:   str = "arena run",
    runs_path:     Path = RUNS_PATH,
    registry_path: Path = REGISTRY_PATH,
) -> ArenaResult:
    """
    Run all patterns against the same dataset and rank by score.
    """
    results: list[ExperimentResult] = []

    for pattern in patterns:
        result = run_experiment(
            pattern       = pattern,
            handle        = handle,
            description   = f"{description} | {pattern.name}",
            runs_path     = runs_path,
            registry_path = registry_path,
        )
        results.append(result)

    ranked = sorted(
        results,
        key=lambda r: (r.status != "crash", r.score),
        reverse=True,
    )

    winner = next((r for r in ranked if r.status != "crash"), None)

    return ArenaResult(
        rankings   = ranked,
        winner     = winner,
        dataset_id = handle.meta.name,
    )
