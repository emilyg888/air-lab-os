"""
Playground — run_experiment() is the core loop entry point.

run_experiment(pattern, handle, config) → ExperimentResult

The engine never knows what domain it is in. It calls
pattern.detect(handle), passes the result to evaluate(), and
records everything to runs.json.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from core.evaluation import EvalMetrics, evaluate, load_policy
from core.registry import (
    PatternRegistry,
    RegistryEntry,
    append_run,
    qualify,
    update_registry,
    use_case_from_dataset_id,
)
from datasets.base import DatasetHandle
from patterns.base import PatternHandler, RunResult

RUNS_PATH     = Path("memory/runs.json")
REGISTRY_PATH = Path("registry.json")
MIN_SCORE_IMPROVEMENT = 0.001


@dataclass
class ExperimentResult:
    """Full output of one experiment run. Written to runs.json."""
    pattern:     str                  # short name
    use_case:    str                  # inferred from dataset_id
    domain:      str
    version:     str
    dataset_id:  str
    pattern_path: str
    config:      dict[str, Any]
    metrics:     EvalMetrics | None   # None on crash
    score:       float
    commit:      str
    status:      str                  # "keep" | "discard" | "crash"
    tier:        str                  # tier at time of run
    description: str
    timestamp:   int                  # unix seconds


def _short_commit() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return r.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no-git"


def run_experiment(
    pattern:       PatternHandler,
    handle:        DatasetHandle,
    description:   str = "",
    runs_path:     Path = RUNS_PATH,
    registry_path: Path = REGISTRY_PATH,
) -> ExperimentResult:
    """
    Run one experiment end-to-end.

    Side effects:
        Appends one record to runs.json.
        Rebuilds and saves registry.json.
    """
    commit    = _short_commit()
    timestamp = int(time.time())
    meta      = handle.meta
    use_case  = use_case_from_dataset_id(meta.name)
    qualified = qualify(use_case, pattern.name)

    registry = PatternRegistry.load(runs_path=runs_path, registry_path=registry_path)
    entry    = registry.get(qualified)
    tier     = entry.tier if entry else "scratch"

    try:
        result: RunResult = pattern.detect(handle)

        policy  = load_policy()
        metrics = evaluate(result, handle, policy)
        score   = metrics.score

        current_best = registry.best_score(qualified)
        status = (
            "keep"
            if (
                current_best is None
                or score > current_best + MIN_SCORE_IMPROVEMENT
            )
            else "discard"
        )

        exp = ExperimentResult(
            pattern      = pattern.name,
            use_case     = use_case,
            domain       = meta.domain,
            version      = pattern.version,
            dataset_id   = meta.name,
            pattern_path = _find_pattern_path(pattern.name),
            config       = pattern.describe(),
            metrics      = metrics,
            score        = score,
            commit       = commit,
            status       = status,
            tier         = tier,
            description  = description or f"{pattern.name} v{pattern.version}",
            timestamp    = timestamp,
        )

    except Exception as exc:
        exp = ExperimentResult(
            pattern      = pattern.name,
            use_case     = use_case,
            domain       = meta.domain,
            version      = getattr(pattern, "version", "?"),
            dataset_id   = meta.name,
            pattern_path = _find_pattern_path(pattern.name),
            config       = pattern.describe() if hasattr(pattern, "describe") else {},
            metrics      = None,
            score        = 0.0,
            commit       = commit,
            status       = "crash",
            tier         = tier,
            description  = f"CRASH: {type(exc).__name__}: {exc}",
            timestamp    = timestamp,
        )

    _write_run(exp, runs_path)
    if exp.status != "crash":
        update_registry(
            pattern_name=qualify(exp.use_case, exp.pattern),
            score=exp.score,
            metadata={"dataset": exp.dataset_id, "description": exp.description},
            policy=load_policy(),
            path=registry_path,
        )

    return exp


def _find_pattern_path(name: str) -> str:
    """Locate a pattern file across tier directories."""
    for tier_dir in ["patterns/scratch", "patterns/working", "patterns/stable"]:
        candidate = Path(tier_dir) / f"{name}.py"
        if candidate.exists():
            return str(candidate)
    return f"patterns/scratch/{name}.py"


def _write_run(exp: ExperimentResult, path: Path) -> None:
    """Serialise ExperimentResult and append to runs.json."""
    m = exp.metrics
    run = {
        "pattern":      exp.pattern,
        "use_case":     exp.use_case,
        "domain":       exp.domain,
        "version":      exp.version,
        "dataset_id":   exp.dataset_id,
        "pattern_path": exp.pattern_path,
        "config":       exp.config,
        "score":        exp.score,
        "status":       exp.status,
        "tier":         exp.tier,
        "commit":       exp.commit,
        "timestamp":    exp.timestamp,
        "description":  exp.description,
        "metrics": {
            "primary_metric_value": m.primary_metric_value,
            "explainability_score": m.explainability_score,
            "latency_score":        m.latency_score,
            "cost_score":           m.cost_score,
            "score":                m.score,
            **(m.extra or {}),
        } if m else {},
    }
    append_run(run, path)
