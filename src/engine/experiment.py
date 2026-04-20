"""
Experiment engine — run_experiment() is the core scientific loop entry point.

run_experiment(pattern, dataset, config) → ExperimentResult

The eval window is always the last 20% of rows sorted by transaction_id.
This is fixed and must not be changed — it makes experiments comparable.
"""

from __future__ import annotations
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
import pandas as pd

from src.detectors.base import Detector, DetectionResult
from src.detectors.velocity import VelocityDetector
from src.detectors.logistic import LogisticDetector
from src.engine.evaluate import evaluate, load_policy, EvaluationMetrics
from src.registry import PatternRegistry, RegistryEntry


PATTERN_REGISTRY_MAP: dict[str, type[Detector]] = {
    "velocity": VelocityDetector,
    "logistic": LogisticDetector,
}


@dataclass
class ExperimentResult:
    """
    Full output of a single experiment run.
    Written to results.tsv and used to update the registry.
    """
    pattern: str
    dataset: str
    config: dict
    metrics: EvaluationMetrics
    score: float
    commit: str
    status: str          # "keep" | "discard" | "crash"
    description: str
    timestamp: int       # unix seconds


def _get_short_commit() -> str:
    """Return 7-char git commit hash, or 'no-git' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no-git"


def _load_dataset(dataset_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load dataset and split into train (80%) / eval (20%) by transaction_id order.
    The eval split is fixed — same slice every time for comparability.
    """
    df = pd.read_csv(dataset_path)

    required_cols = {"transaction_id", "amount", "user_id", "timestamp", "merchant_id", "is_fraud"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    df = df.sort_values("transaction_id").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    eval_df = df.iloc[split_idx:].copy()
    return train_df, eval_df


def run_experiment(
    pattern: str,
    dataset: str,
    config: dict | None = None,
    description: str = "",
    registry_path: Path = Path("registry.json"),
    results_path: Path = Path("results.tsv"),
) -> ExperimentResult:
    """
    Run one experiment end-to-end.

    Args:
        pattern:       name of the detector to run (must be in PATTERN_REGISTRY_MAP)
        dataset:       path to CSV file with transaction data
        config:        hyperparameter overrides passed to the detector constructor
        description:   short human-readable description for the TSV log
        registry_path: path to registry.json
        results_path:  path to results.tsv

    Returns:
        ExperimentResult with full metrics and score

    Side effects:
        - Appends one row to results.tsv
        - Updates registry.json via PatternRegistry
    """
    config = config or {}
    commit = _get_short_commit()
    ts = int(time.time())

    if pattern not in PATTERN_REGISTRY_MAP:
        exp_result = ExperimentResult(
            pattern=pattern,
            dataset=dataset,
            config=config,
            metrics=None,
            score=0.0,
            commit=commit,
            status="crash",
            description=f"CRASH: unknown pattern '{pattern}'",
            timestamp=ts,
        )
        _append_tsv(exp_result, results_path)
        registry = PatternRegistry.load(registry_path, results_path)
        registry.save(registry_path)
        return exp_result

    try:
        # --- Build detector ---
        detector_cls = PATTERN_REGISTRY_MAP[pattern]
        detector = detector_cls(**config)

        # --- Load data ---
        train_df, eval_df = _load_dataset(dataset)

        # --- Train if needed ---
        if hasattr(detector, "fit"):
            detector.fit(train_df)

        # --- Run on eval split ---
        result: DetectionResult = detector.detect(eval_df)

        # --- Score ---
        policy = load_policy()
        labels = eval_df["is_fraud"].astype(bool).tolist()
        metrics = evaluate(result, labels, policy)
        score = metrics.score

        # --- Determine status ---
        registry = PatternRegistry.load(registry_path, results_path)
        current_best = registry.best_score(pattern)
        status = "keep" if (current_best is None or score > current_best) else "discard"

        exp_result = ExperimentResult(
            pattern=pattern,
            dataset=dataset,
            config=config,
            metrics=metrics,
            score=score,
            commit=commit,
            status=status,
            description=description or f"{pattern} config={config}",
            timestamp=ts,
        )

    except Exception as e:
        # Crash — log it and return a zeroed result
        exp_result = ExperimentResult(
            pattern=pattern,
            dataset=dataset,
            config=config,
            metrics=None,
            score=0.0,
            commit=commit,
            status="crash",
            description=f"CRASH: {type(e).__name__}: {e}",
            timestamp=ts,
        )

    # --- Write to TSV ---
    _append_tsv(exp_result, results_path)

    # --- Update registry ---
    registry = PatternRegistry.load(registry_path, results_path)
    registry.save(registry_path)

    return exp_result


def _append_tsv(result: ExperimentResult, path: Path) -> None:
    """Append one row to results.tsv. Creates the file with header if missing."""
    header = "commit\tpattern\tscore\tprecision\trecall\tf1\texpl_score\t" \
             "latency_ms\tcost_per_1k\tstatus\tdescription\ttimestamp\n"

    if not path.exists():
        path.write_text(header)

    m = result.metrics
    if m is not None:
        row = (
            f"{result.commit}\t{result.pattern}\t{result.score:.6f}\t"
            f"{m.precision:.4f}\t{m.recall:.4f}\t{m.f1:.4f}\t"
            f"{m.explainability_score:.4f}\t"
            f"{result.metrics.latency_score:.4f}\t0.0000\t"
            f"{result.status}\t{result.description}\t{result.timestamp}\n"
        )
    else:
        row = (
            f"{result.commit}\t{result.pattern}\t0.000000\t"
            f"0.0000\t0.0000\t0.0000\t0.0000\t0.0000\t0.0000\t"
            f"{result.status}\t{result.description}\t{result.timestamp}\n"
        )

    with open(path, "a") as f:
        f.write(row)


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description="Run one experiment")
    parser.add_argument("--pattern", required=True, choices=list(PATTERN_REGISTRY_MAP.keys()))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default="{}", help="JSON config overrides")
    parser.add_argument("--description", default="")
    args = parser.parse_args()

    result = run_experiment(
        pattern=args.pattern,
        dataset=args.dataset,
        config=json.loads(args.config),
        description=args.description,
    )
    print(f"\n--- Experiment result ---")
    print(f"pattern:  {result.pattern}")
    print(f"status:   {result.status}")
    print(f"score:    {result.score:.6f}")
    if result.metrics:
        print(f"f1:       {result.metrics.f1:.4f}")
        print(f"precision:{result.metrics.precision:.4f}")
        print(f"recall:   {result.metrics.recall:.4f}")
        print(f"expl:     {result.metrics.explainability_score:.4f}")
    print(f"commit:   {result.commit}")
    print(f"description: {result.description}")
