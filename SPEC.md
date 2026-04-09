# SPEC.md — Phase 1: Foundation

> **For Claude Code.** Read this file in full before writing any code.
> Build everything in this spec, in the order listed. Do not proceed to
> the next section until the current section's deliverables pass.

---

## Context

This is a fraud detection pattern research system inspired by Karpathy's
autoresearch repo. The goal is an autonomous experiment loop that runs
detectors against transaction data, scores them with a locked evaluation
policy, and tracks results in a registry.

Phase 1 builds the skeleton: one experiment runs end-to-end and writes
its result to the registry. No UI, no QWEN planner, no arena yet.

---

## Repo layout

Create this directory structure exactly:

```
fraud-engine/
  CLAUDE.md                   ← standing instructions (content below)
  SPEC.md                     ← this file
  scoring_policy.yaml         ← locked scoring weights (content below)
  results.tsv                 ← experiment log (git-tracked, appended only)
  registry.json               ← derived state (rebuilt on startup)
  pyproject.toml              ← dependencies
  src/
    __init__.py
    registry.py               ← PatternRegistry class
    detectors/
      __init__.py
      base.py                 ← Detector ABC + DetectionResult
      velocity.py             ← VelocityDetector
      logistic.py             ← LogisticDetector
    engine/
      __init__.py
      experiment.py           ← run_experiment()
      evaluate.py             ← evaluate() + load_policy()
  tests/
    test_detectors.py
    test_experiment.py
    test_evaluate.py
    test_registry.py
  data/
    sample_transactions.csv   ← synthetic sample (generate this)
```

---

## CLAUDE.md

Write this content exactly to `CLAUDE.md`:

```markdown
# CLAUDE.md — fraud-engine

Standing instructions for all Claude Code sessions in this repo.
Read this before reading SPEC.md or any other file.

## What this repo is

An autonomous fraud detection research system. An experiment loop runs
detectors against transaction data, scores them with a locked evaluation
policy, and promotes the best patterns through bronze → silver → gold.

## Rules — never break these

1. **Never modify `scoring_policy.yaml`.**
   It is the locked evaluation contract. The evaluate() function reads it.
   No other component writes to it. If you think a weight should change,
   surface it as a comment to the human — do not edit the file.

2. **`registry.json` is derived, not authoritative.**
   It is rebuilt from `results.tsv` on every startup. If they conflict,
   `results.tsv` wins. Never write registry.json directly — only write
   it via `PatternRegistry.save()` after rebuilding from the TSV.

3. **`results.tsv` is append-only.**
   Never delete rows. Never modify existing rows. Only append new rows
   at the end. This file is the permanent experiment record.

4. **The eval window is fixed.**
   `run_experiment()` always evaluates on the same dataset slice:
   the last 20% of rows in the dataset file, sorted by transaction_id.
   Never change this split logic — it makes experiments comparable.

5. **Run tests before every commit.**
   `uv run pytest tests/ -q` must pass with zero failures.
   If tests fail, fix before committing.

6. **One commit per experiment.**
   When an experiment result is kept (score improves), commit with:
   `git commit -m "exp: <pattern_name> score=<score>"`
   When discarded: `git reset --soft HEAD~1`

7. **Never install packages not in pyproject.toml.**
   If you need a new dependency, stop and ask the human.

## File ownership

| File | Owner | Claude Code may… |
|------|-------|-----------------|
| scoring_policy.yaml | Human | Read only |
| CLAUDE.md | Human | Read only |
| SPEC.md | Human | Read only |
| results.tsv | System | Append only |
| registry.json | System | Overwrite via PatternRegistry.save() |
| src/** | Claude Code | Read + write |
| tests/** | Claude Code | Read + write |

## How to run

```bash
uv run python -m src.engine.experiment --pattern velocity --dataset data/sample_transactions.csv
uv run pytest tests/ -q
```

## Session startup checklist

1. Read CLAUDE.md (this file)
2. Read the current SPEC.md
3. Run `uv run pytest tests/ -q` to confirm baseline
4. Check `results.tsv` for experiment history
5. Check `registry.json` for current registry state
```

---

## scoring_policy.yaml

Write this content exactly to `scoring_policy.yaml`.
This file is locked — do not modify it during the build:

```yaml
# Fraud detection scoring policy
# Locked — do not modify. Human-owned.
# Weights must sum to 1.0.

version: "1.0"

weights:
  precision_recall: 0.40   # F1-weighted blend of precision and recall
  explainability:   0.25   # fraction of flags with non-empty explanation string
  latency:          0.20   # inverse-scaled: 0ms=1.0, 200ms=0.0, linear
  cost:             0.15   # inverse-scaled: $0=1.0, $0.10/1k=0.0, linear

promotion:
  silver_threshold: 0.65   # minimum score to achieve silver status
  gold_threshold:   0.78   # minimum score to become promotion candidate
  min_runs:         3      # minimum experiment runs before promotion eligible

latency:
  max_ms: 200              # latency at which latency_score = 0.0

cost:
  max_per_1k: 0.10         # cost (USD per 1000 transactions) at which cost_score = 0.0
```

---

## pyproject.toml

```toml
[project]
name = "fraud-engine"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "pandas>=2.0",
    "scikit-learn>=1.3",
    "pyyaml>=6.0",
    "pytest>=8.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

---

## Component 1: Detector interface

### `src/detectors/base.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class DetectionResult:
    """
    Output contract for every detector.

    All three lists must be the same length as the input DataFrame.
    flags       — True if transaction is flagged as fraud
    scores      — float in [0.0, 1.0], fraud probability or confidence
    explanation — human-readable reason string, or "" if none
    latency_ms  — wall-clock milliseconds for the full run() call
    cost_per_1k — estimated USD cost per 1000 transactions (0.0 for local models)
    """
    flags: list[bool]
    scores: list[float]
    explanation: list[str]
    latency_ms: float = 0.0
    cost_per_1k: float = 0.0

    def __post_init__(self):
        n = len(self.flags)
        assert len(self.scores) == n, "scores length must match flags"
        assert len(self.explanation) == n, "explanation length must match flags"
        assert all(0.0 <= s <= 1.0 for s in self.scores), "scores must be in [0, 1]"


class Detector(ABC):
    """
    Abstract base for all fraud detectors.

    Subclasses implement run(). The base class wraps run() to
    record wall-clock latency automatically.

    Usage:
        detector = VelocityDetector(window_seconds=3600, threshold=5)
        result = detector.detect(df)
    """

    name: str = "base"

    def detect(self, data: pd.DataFrame) -> DetectionResult:
        """
        Public entry point. Times the call and injects latency_ms.
        Do not override — override run() instead.
        """
        import time
        t0 = time.perf_counter()
        result = self.run(data)
        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    @abstractmethod
    def run(self, data: pd.DataFrame) -> DetectionResult:
        """
        Implement fraud detection logic here.

        Args:
            data: DataFrame with at least these columns:
                  transaction_id (str), amount (float),
                  user_id (str), timestamp (int, unix seconds),
                  merchant_id (str)

        Returns:
            DetectionResult with flags, scores, explanation lists
            the same length as len(data).
            Do NOT set latency_ms — the base class sets it.
        """
        ...

    def describe(self) -> dict:
        """Return config dict for logging. Override to add hyperparams."""
        return {"detector": self.name}
```

### `src/detectors/velocity.py`

```python
import pandas as pd
from .base import Detector, DetectionResult


class VelocityDetector(Detector):
    """
    Flags transactions where a user exceeds `threshold` transactions
    within a rolling `window_seconds` window.

    Explainability: every flag includes the count and window used.

    Args:
        window_seconds: rolling window size in seconds (default 3600 = 1h)
        threshold:      max transactions in window before flagging (default 5)
        score_cap:      maximum fraud score assigned (default 0.9)
    """

    name = "velocity_detector"

    def __init__(
        self,
        window_seconds: int = 3600,
        threshold: int = 5,
        score_cap: float = 0.9,
    ):
        self.window_seconds = window_seconds
        self.threshold = threshold
        self.score_cap = score_cap

    def run(self, data: pd.DataFrame) -> DetectionResult:
        df = data.copy().reset_index(drop=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

        flags = [False] * len(df)
        scores = [0.0] * len(df)
        explanation = [""] * len(df)

        for idx, row in df.iterrows():
            window_start = row["timestamp"] - self.window_seconds
            user_txns = df[
                (df["user_id"] == row["user_id"])
                & (df["timestamp"] >= window_start)
                & (df["timestamp"] <= row["timestamp"])
            ]
            count = len(user_txns)
            if count > self.threshold:
                flags[idx] = True
                # Score scales linearly with excess, capped at score_cap
                excess = count - self.threshold
                scores[idx] = min(self.score_cap, 0.5 + 0.1 * excess)
                explanation[idx] = (
                    f"{count} transactions in {self.window_seconds}s window "
                    f"(threshold={self.threshold})"
                )

        return DetectionResult(
            flags=flags,
            scores=scores,
            explanation=explanation,
            cost_per_1k=0.0,
        )

    def describe(self) -> dict:
        return {
            "detector": self.name,
            "window_seconds": self.window_seconds,
            "threshold": self.threshold,
            "score_cap": self.score_cap,
        }
```

### `src/detectors/logistic.py`

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from .base import Detector, DetectionResult


class LogisticDetector(Detector):
    """
    Logistic regression baseline detector.

    Features: amount, hour_of_day, txn_count_24h (per user),
              amount_zscore (per user rolling).

    Trains on the first 80% of rows (by transaction_id order),
    predicts on whatever data is passed to run().

    Explainability: top contributing feature per prediction.

    Args:
        C: regularisation strength (default 1.0)
        max_iter: solver iterations (default 200)
        threshold: probability threshold for flagging (default 0.5)
    """

    name = "logistic_detector"

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 200,
        threshold: float = 0.5,
    ):
        self.C = C
        self.max_iter = max_iter
        self.threshold = threshold
        self._model: LogisticRegression | None = None
        self._scaler: StandardScaler | None = None
        self._feature_names: list[str] = []

    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Train on labelled data. Requires an `is_fraud` column (bool/int).
        Called automatically by run() if the model is not yet fitted.
        """
        X, _ = self._build_features(train_data)
        y = train_data["is_fraud"].astype(int).values

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._model = LogisticRegression(
            C=self.C, max_iter=self.max_iter, random_state=42
        )
        self._model.fit(X_scaled, y)

    def run(self, data: pd.DataFrame) -> DetectionResult:
        if self._model is None:
            raise RuntimeError(
                "LogisticDetector must be fitted before calling run(). "
                "Call detector.fit(train_df) first, or use run_experiment() "
                "which handles the train/eval split automatically."
            )

        X, _ = self._build_features(data)
        X_scaled = self._scaler.transform(X)
        probs = self._model.predict_proba(X_scaled)[:, 1]
        flags = (probs >= self.threshold).tolist()
        scores = probs.tolist()

        # Explainability: name the top contributing feature per prediction
        coefs = self._model.coef_[0]
        explanation = []
        for i, row_x in enumerate(X_scaled):
            contributions = coefs * row_x
            top_idx = int(np.argmax(np.abs(contributions)))
            top_feat = self._feature_names[top_idx]
            direction = "high" if contributions[top_idx] > 0 else "low"
            explanation.append(
                f"top feature: {top_feat} ({direction}, "
                f"coef={coefs[top_idx]:.3f})"
            )

        return DetectionResult(
            flags=flags,
            scores=[float(s) for s in scores],
            explanation=explanation,
            cost_per_1k=0.0,
        )

    def _build_features(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        feats = pd.DataFrame(index=df.index)
        feats["amount"] = df["amount"].fillna(0.0)
        feats["hour_of_day"] = (df["timestamp"] % 86400 / 3600).fillna(0.0)

        # Transactions per user in last 24h
        txn_counts = []
        for _, row in df.iterrows():
            window = df[
                (df["user_id"] == row["user_id"])
                & (df["timestamp"] >= row["timestamp"] - 86400)
                & (df["timestamp"] <= row["timestamp"])
            ]
            txn_counts.append(len(window))
        feats["txn_count_24h"] = txn_counts

        # Amount z-score per user
        user_means = df.groupby("user_id")["amount"].transform("mean")
        user_stds = df.groupby("user_id")["amount"].transform("std").fillna(1.0)
        feats["amount_zscore"] = ((df["amount"] - user_means) / user_stds).fillna(0.0)

        self._feature_names = list(feats.columns)
        return feats.values, self._feature_names

    def describe(self) -> dict:
        return {
            "detector": self.name,
            "C": self.C,
            "max_iter": self.max_iter,
            "threshold": self.threshold,
        }
```

---

## Component 2: Evaluation engine

### `src/engine/evaluate.py`

```python
"""
Evaluation engine — locked scoring layer.

evaluate(result, labels) → float in [0.0, 1.0]

Reads weights from scoring_policy.yaml. Do not hardcode weights here.
The policy file is the single source of truth.
"""

from __future__ import annotations
import yaml
from pathlib import Path
from dataclasses import dataclass
from sklearn.metrics import precision_score, recall_score, f1_score
from src.detectors.base import DetectionResult


POLICY_PATH = Path(__file__).parent.parent.parent / "scoring_policy.yaml"


@dataclass
class ScoringPolicy:
    precision_recall_weight: float
    explainability_weight: float
    latency_weight: float
    cost_weight: float
    latency_max_ms: float
    cost_max_per_1k: float
    silver_threshold: float
    gold_threshold: float
    min_runs: int

    def validate(self) -> None:
        total = (
            self.precision_recall_weight
            + self.explainability_weight
            + self.latency_weight
            + self.cost_weight
        )
        assert abs(total - 1.0) < 1e-6, (
            f"Scoring weights must sum to 1.0, got {total:.4f}. "
            f"Check scoring_policy.yaml."
        )


def load_policy(path: Path = POLICY_PATH) -> ScoringPolicy:
    """Load and validate the scoring policy. Raises on missing file or bad weights."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    w = raw["weights"]
    policy = ScoringPolicy(
        precision_recall_weight=w["precision_recall"],
        explainability_weight=w["explainability"],
        latency_weight=w["latency"],
        cost_weight=w["cost"],
        latency_max_ms=raw["latency"]["max_ms"],
        cost_max_per_1k=raw["cost"]["max_per_1k"],
        silver_threshold=raw["promotion"]["silver_threshold"],
        gold_threshold=raw["promotion"]["gold_threshold"],
        min_runs=raw["promotion"]["min_runs"],
    )
    policy.validate()
    return policy


@dataclass
class EvaluationMetrics:
    """Full breakdown of the evaluation. Score is the weighted composite."""
    precision: float
    recall: float
    f1: float
    precision_recall_score: float   # normalised [0,1], same as f1 here
    explainability_score: float     # fraction of flags with non-empty explanation
    latency_score: float            # 1.0 = 0ms, 0.0 = max_ms
    cost_score: float               # 1.0 = free, 0.0 = max cost
    score: float                    # weighted composite


def evaluate(
    result: DetectionResult,
    labels: list[bool],
    policy: ScoringPolicy | None = None,
) -> EvaluationMetrics:
    """
    Score a DetectionResult against ground truth labels.

    Args:
        result: output of Detector.detect()
        labels: ground truth fraud flags, same length as result.flags
        policy: loaded ScoringPolicy; loads from yaml if None

    Returns:
        EvaluationMetrics with per-dimension scores and weighted composite
    """
    if policy is None:
        policy = load_policy()

    assert len(result.flags) == len(labels), (
        f"result has {len(result.flags)} flags but labels has {len(labels)}"
    )

    # --- Precision / recall ---
    y_true = [int(l) for l in labels]
    y_pred = [int(f) for f in result.flags]

    if sum(y_pred) == 0:
        # Nothing flagged — precision undefined, recall = 0
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

    pr_score = float(f1)  # use F1 as the precision/recall composite

    # --- Explainability ---
    flagged_indices = [i for i, f in enumerate(result.flags) if f]
    if flagged_indices:
        with_explanation = sum(
            1 for i in flagged_indices if result.explanation[i].strip()
        )
        expl_score = with_explanation / len(flagged_indices)
    else:
        expl_score = 1.0  # nothing flagged = nothing to explain

    # --- Latency ---
    latency_score = max(
        0.0, 1.0 - result.latency_ms / policy.latency_max_ms
    )

    # --- Cost ---
    cost_score = max(
        0.0, 1.0 - result.cost_per_1k / policy.cost_max_per_1k
    )

    # --- Weighted composite ---
    composite = (
        policy.precision_recall_weight * pr_score
        + policy.explainability_weight * expl_score
        + policy.latency_weight * latency_score
        + policy.cost_weight * cost_score
    )

    return EvaluationMetrics(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        precision_recall_score=round(pr_score, 4),
        explainability_score=round(expl_score, 4),
        latency_score=round(latency_score, 4),
        cost_score=round(cost_score, 4),
        score=round(composite, 4),
    )
```

---

## Component 3: Experiment engine

### `src/engine/experiment.py`

```python
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
        raise ValueError(
            f"Unknown pattern '{pattern}'. "
            f"Available: {list(PATTERN_REGISTRY_MAP.keys())}"
        )

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
```

---

## Component 4: Pattern registry

### `src/registry.py`

```python
"""
Pattern registry — live system state rebuilt from results.tsv.

PatternRegistry is rebuilt from the TSV on every startup.
Never write registry.json directly — always go through PatternRegistry.save().

Status values:
  bronze  — score >= 0 but below silver_threshold
  silver  — score >= silver_threshold
  gold    — manually promoted (promotion_candidate surfaced to human)
  running — currently being evaluated
  discard — ran but did not improve
  crash   — experiment crashed
"""

from __future__ import annotations
import json
import csv
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class RegistryEntry:
    pattern: str
    status: str                     # bronze | silver | gold | running | discard | crash
    confidence: float               # best score achieved
    runs: int                       # total experiment count
    last_commit: str
    last_updated: int               # unix timestamp
    promotion_candidate: bool       # True if score >= gold_threshold and runs >= min_runs
    best_precision: float = 0.0
    best_recall: float = 0.0
    best_f1: float = 0.0
    description: str = ""


class PatternRegistry:
    """
    Manages pattern state. Rebuilt from results.tsv on load.

    Usage:
        registry = PatternRegistry.load(registry_path, results_path)
        entry = registry.get("velocity")
        registry.save(registry_path)
    """

    # These thresholds mirror scoring_policy.yaml.
    # They are read from the policy file on load — do not hardcode here.
    SILVER_THRESHOLD: float = 0.65
    GOLD_THRESHOLD: float = 0.78
    MIN_RUNS: int = 3

    def __init__(self, entries: dict[str, RegistryEntry]):
        self._entries = entries

    @classmethod
    def load(
        cls,
        registry_path: Path,
        results_path: Path,
    ) -> "PatternRegistry":
        """
        Rebuild registry from results.tsv.
        Ignores registry.json — TSV is authoritative.
        """
        # Load promotion thresholds from scoring policy
        try:
            from src.engine.evaluate import load_policy
            policy = load_policy()
            cls.SILVER_THRESHOLD = policy.silver_threshold
            cls.GOLD_THRESHOLD = policy.gold_threshold
            cls.MIN_RUNS = policy.min_runs
        except Exception:
            pass  # use class defaults if policy unavailable

        # Load existing gold statuses from registry.json (only gold is preserved)
        gold_patterns: set[str] = set()
        if registry_path.exists():
            try:
                data = json.loads(registry_path.read_text())
                gold_patterns = {
                    k for k, v in data.items()
                    if isinstance(v, dict) and v.get("status") == "gold"
                }
            except Exception:
                pass

        entries: dict[str, RegistryEntry] = {}

        if not results_path.exists():
            return cls(entries)

        with open(results_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                pattern = row["pattern"]
                score = float(row.get("score", 0))
                status = row.get("status", "discard")
                ts = int(row.get("timestamp", 0))

                if pattern not in entries:
                    entries[pattern] = RegistryEntry(
                        pattern=pattern,
                        status="bronze",
                        confidence=0.0,
                        runs=0,
                        last_commit=row.get("commit", ""),
                        last_updated=ts,
                        promotion_candidate=False,
                        description=row.get("description", ""),
                    )

                e = entries[pattern]
                e.runs += 1
                e.last_updated = max(e.last_updated, ts)
                e.last_commit = row.get("commit", e.last_commit)

                if status in ("keep",) and score > e.confidence:
                    e.confidence = score
                    e.best_precision = float(row.get("precision", 0))
                    e.best_recall = float(row.get("recall", 0))
                    e.best_f1 = float(row.get("f1", 0))

        # Compute statuses
        for pattern, e in entries.items():
            if pattern in gold_patterns:
                e.status = "gold"
            elif e.confidence >= cls.SILVER_THRESHOLD:
                e.status = "silver"
            else:
                e.status = "bronze"

            e.promotion_candidate = (
                e.confidence >= cls.GOLD_THRESHOLD
                and e.runs >= cls.MIN_RUNS
                and e.status != "gold"
            )

        return cls(entries)

    def get(self, pattern: str) -> Optional[RegistryEntry]:
        return self._entries.get(pattern)

    def all(self) -> list[RegistryEntry]:
        return sorted(self._entries.values(), key=lambda e: e.confidence, reverse=True)

    def best_score(self, pattern: str) -> Optional[float]:
        e = self._entries.get(pattern)
        return e.confidence if e else None

    def promotion_candidates(self) -> list[RegistryEntry]:
        return [e for e in self._entries.values() if e.promotion_candidate]

    def save(self, path: Path) -> None:
        """Write registry.json. Derived from TSV — safe to overwrite."""
        data = {pattern: asdict(entry) for pattern, entry in self._entries.items()}
        path.write_text(json.dumps(data, indent=2))
```

---

## Sample dataset

### `data/sample_transactions.csv` — generate this file

Write a Python script `data/generate_sample.py` that creates
`data/sample_transactions.csv` with these properties:

- 2000 rows
- Columns: `transaction_id` (str, T0001–T2000), `amount` (float, 1–5000),
  `user_id` (str, U001–U050), `timestamp` (int, unix seconds, spread over 7 days),
  `merchant_id` (str, M001–M020), `is_fraud` (bool)
- Fraud rate: ~8% of rows
- Fraud signal: users with >6 transactions in any 1-hour window are more
  likely to be fraud (velocity signal for VelocityDetector to detect)
- Higher amounts (>$1000) slightly more likely to be fraud
  (feature signal for LogisticDetector)

Run `uv run python data/generate_sample.py` to produce the CSV.

---

## Tests

### `tests/test_detectors.py`

```python
import pytest
import pandas as pd
from src.detectors.base import DetectionResult
from src.detectors.velocity import VelocityDetector
from src.detectors.logistic import LogisticDetector


@pytest.fixture
def sample_df():
    """Minimal DataFrame satisfying the detector input contract."""
    return pd.DataFrame({
        "transaction_id": ["T001", "T002", "T003", "T004", "T005", "T006", "T007"],
        "amount": [100.0, 200.0, 50.0, 300.0, 150.0, 80.0, 500.0],
        "user_id": ["U1", "U1", "U1", "U1", "U1", "U1", "U2"],
        "timestamp": [0, 100, 200, 300, 400, 500, 600],
        "merchant_id": ["M1"] * 7,
        "is_fraud": [False, False, False, False, False, True, False],
    })


def test_velocity_result_shape(sample_df):
    d = VelocityDetector(window_seconds=1000, threshold=3)
    r = d.detect(sample_df)
    assert len(r.flags) == len(sample_df)
    assert len(r.scores) == len(sample_df)
    assert len(r.explanation) == len(sample_df)


def test_velocity_flags_high_frequency(sample_df):
    d = VelocityDetector(window_seconds=1000, threshold=3)
    r = d.detect(sample_df)
    # U1 has 6 txns in window of 1000s — should get flagged
    u1_flags = [r.flags[i] for i, uid in enumerate(sample_df["user_id"]) if uid == "U1"]
    assert any(u1_flags), "U1 exceeds threshold and should be flagged"


def test_velocity_explanation_populated(sample_df):
    d = VelocityDetector(window_seconds=1000, threshold=3)
    r = d.detect(sample_df)
    for i, flag in enumerate(r.flags):
        if flag:
            assert r.explanation[i] != "", f"Flagged transaction {i} has no explanation"


def test_velocity_scores_in_range(sample_df):
    d = VelocityDetector()
    r = d.detect(sample_df)
    assert all(0.0 <= s <= 1.0 for s in r.scores)


def test_logistic_requires_fit(sample_df):
    d = LogisticDetector()
    with pytest.raises(RuntimeError, match="fitted"):
        d.run(sample_df)


def test_logistic_fit_and_run(sample_df):
    d = LogisticDetector()
    d.fit(sample_df)
    r = d.detect(sample_df)
    assert len(r.flags) == len(sample_df)
    assert all(0.0 <= s <= 1.0 for s in r.scores)


def test_logistic_explanations_populated(sample_df):
    d = LogisticDetector()
    d.fit(sample_df)
    r = d.detect(sample_df)
    assert all(e != "" for e in r.explanation), "All predictions should have explanations"


def test_latency_recorded(sample_df):
    d = VelocityDetector()
    r = d.detect(sample_df)
    assert r.latency_ms > 0, "latency_ms should be set by base class detect()"
```

### `tests/test_evaluate.py`

```python
import pytest
from src.detectors.base import DetectionResult
from src.engine.evaluate import evaluate, load_policy, ScoringPolicy


@pytest.fixture
def policy():
    return load_policy()


def perfect_result(n=10):
    return DetectionResult(
        flags=[True] * 3 + [False] * 7,
        scores=[0.9] * 3 + [0.1] * 7,
        explanation=["high velocity"] * 3 + [""] * 7,
        latency_ms=10.0,
        cost_per_1k=0.0,
    )


def perfect_labels():
    return [True] * 3 + [False] * 7


def test_evaluate_returns_metrics(policy):
    r = perfect_result()
    m = evaluate(r, perfect_labels(), policy)
    assert 0.0 <= m.score <= 1.0


def test_perfect_precision_recall(policy):
    r = perfect_result()
    m = evaluate(r, perfect_labels(), policy)
    assert m.precision == pytest.approx(1.0)
    assert m.recall == pytest.approx(1.0)
    assert m.f1 == pytest.approx(1.0)


def test_explainability_score(policy):
    r = perfect_result()
    m = evaluate(r, perfect_labels(), policy)
    assert m.explainability_score == pytest.approx(1.0)


def test_no_flags_gives_zero_pr(policy):
    r = DetectionResult(
        flags=[False] * 10,
        scores=[0.0] * 10,
        explanation=[""] * 10,
        latency_ms=5.0,
        cost_per_1k=0.0,
    )
    m = evaluate(r, perfect_labels(), policy)
    assert m.f1 == 0.0


def test_latency_score_decreases_with_latency(policy):
    fast = DetectionResult(flags=[False]*5, scores=[0.0]*5, explanation=[""]*5, latency_ms=0.0)
    slow = DetectionResult(flags=[False]*5, scores=[0.0]*5, explanation=[""]*5, latency_ms=200.0)
    labels = [False] * 5
    m_fast = evaluate(fast, labels, policy)
    m_slow = evaluate(slow, labels, policy)
    assert m_fast.latency_score > m_slow.latency_score


def test_weights_sum_to_one(policy):
    total = (
        policy.precision_recall_weight
        + policy.explainability_weight
        + policy.latency_weight
        + policy.cost_weight
    )
    assert abs(total - 1.0) < 1e-6
```

### `tests/test_experiment.py`

```python
import pytest
import tempfile
from pathlib import Path
import pandas as pd
from src.engine.experiment import run_experiment, _load_dataset


@pytest.fixture
def tmp_paths(tmp_path):
    return {
        "registry": tmp_path / "registry.json",
        "results": tmp_path / "results.tsv",
    }


@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame({
        "transaction_id": [f"T{i:04d}" for i in range(100)],
        "amount": [float(i * 10) for i in range(100)],
        "user_id": [f"U{i % 5:02d}" for i in range(100)],
        "timestamp": list(range(0, 10000, 100)),
        "merchant_id": [f"M{i % 3:02d}" for i in range(100)],
        "is_fraud": [i % 12 == 0 for i in range(100)],
    })
    p = tmp_path / "txns.csv"
    df.to_csv(p, index=False)
    return str(p)


def test_run_experiment_velocity(sample_csv, tmp_paths):
    result = run_experiment(
        pattern="velocity",
        dataset=sample_csv,
        registry_path=tmp_paths["registry"],
        results_path=tmp_paths["results"],
        description="test run",
    )
    assert result.pattern == "velocity"
    assert result.status in ("keep", "discard", "crash")
    assert 0.0 <= result.score <= 1.0


def test_run_experiment_writes_tsv(sample_csv, tmp_paths):
    run_experiment(
        pattern="velocity",
        dataset=sample_csv,
        registry_path=tmp_paths["registry"],
        results_path=tmp_paths["results"],
    )
    assert tmp_paths["results"].exists()
    lines = tmp_paths["results"].read_text().strip().split("\n")
    assert len(lines) == 2  # header + 1 result row


def test_run_experiment_updates_registry(sample_csv, tmp_paths):
    run_experiment(
        pattern="velocity",
        dataset=sample_csv,
        registry_path=tmp_paths["registry"],
        results_path=tmp_paths["results"],
    )
    assert tmp_paths["registry"].exists()
    import json
    data = json.loads(tmp_paths["registry"].read_text())
    assert "velocity" in data


def test_eval_split_is_fixed(sample_csv):
    train, eval_df = _load_dataset(sample_csv)
    assert len(train) == 80
    assert len(eval_df) == 20
    # Run twice — same split
    train2, eval2 = _load_dataset(sample_csv)
    assert list(train["transaction_id"]) == list(train2["transaction_id"])
    assert list(eval_df["transaction_id"]) == list(eval2["transaction_id"])


def test_unknown_pattern_raises(sample_csv, tmp_paths):
    result = run_experiment(
        pattern="nonexistent",
        dataset=sample_csv,
        registry_path=tmp_paths["registry"],
        results_path=tmp_paths["results"],
    )
    assert result.status == "crash"
```

### `tests/test_registry.py`

```python
import pytest
import json
import tempfile
from pathlib import Path
from src.registry import PatternRegistry


TSV_HEADER = "commit\tpattern\tscore\tprecision\trecall\tf1\texpl_score\tlatency_ms\tcost_per_1k\tstatus\tdescription\ttimestamp\n"


def write_tsv(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        f.write(TSV_HEADER)
        for r in rows:
            f.write(
                f"{r['commit']}\t{r['pattern']}\t{r['score']:.6f}\t"
                f"{r.get('precision',0):.4f}\t{r.get('recall',0):.4f}\t"
                f"{r.get('f1',0):.4f}\t{r.get('expl_score',0):.4f}\t"
                f"0.0000\t0.0000\t{r['status']}\t{r.get('description','')}\t"
                f"{r.get('timestamp',0)}\n"
            )


def test_registry_loads_from_tsv(tmp_path):
    results = tmp_path / "results.tsv"
    registry = tmp_path / "registry.json"
    write_tsv(results, [
        {"commit": "abc1234", "pattern": "velocity", "score": 0.74,
         "f1": 0.74, "status": "keep", "timestamp": 1000},
    ])
    reg = PatternRegistry.load(registry, results)
    entry = reg.get("velocity")
    assert entry is not None
    assert entry.confidence == pytest.approx(0.74)
    assert entry.runs == 1


def test_registry_status_silver(tmp_path):
    results = tmp_path / "results.tsv"
    registry = tmp_path / "registry.json"
    write_tsv(results, [
        {"commit": "abc1234", "pattern": "velocity", "score": 0.70,
         "f1": 0.70, "status": "keep", "timestamp": 1000},
    ])
    reg = PatternRegistry.load(registry, results)
    assert reg.get("velocity").status == "silver"


def test_registry_status_bronze(tmp_path):
    results = tmp_path / "results.tsv"
    registry = tmp_path / "registry.json"
    write_tsv(results, [
        {"commit": "abc1234", "pattern": "velocity", "score": 0.50,
         "f1": 0.50, "status": "keep", "timestamp": 1000},
    ])
    reg = PatternRegistry.load(registry, results)
    assert reg.get("velocity").status == "bronze"


def test_registry_promotion_candidate(tmp_path):
    results = tmp_path / "results.tsv"
    registry = tmp_path / "registry.json"
    rows = [
        {"commit": f"abc{i}", "pattern": "velocity", "score": 0.80,
         "f1": 0.80, "status": "keep", "timestamp": i}
        for i in range(3)
    ]
    write_tsv(results, rows)
    reg = PatternRegistry.load(registry, results)
    assert reg.get("velocity").promotion_candidate is True


def test_registry_save_and_reload(tmp_path):
    results = tmp_path / "results.tsv"
    registry_path = tmp_path / "registry.json"
    write_tsv(results, [
        {"commit": "abc1234", "pattern": "logistic", "score": 0.69,
         "f1": 0.69, "status": "keep", "timestamp": 1000},
    ])
    reg = PatternRegistry.load(registry_path, results)
    reg.save(registry_path)
    assert registry_path.exists()
    data = json.loads(registry_path.read_text())
    assert "logistic" in data
    assert data["logistic"]["confidence"] == pytest.approx(0.69)


def test_registry_gold_preserved_across_rebuild(tmp_path):
    results = tmp_path / "results.tsv"
    registry_path = tmp_path / "registry.json"
    write_tsv(results, [
        {"commit": "abc1234", "pattern": "velocity", "score": 0.81,
         "f1": 0.81, "status": "keep", "timestamp": 1000},
    ])
    # Manually set gold in registry.json
    registry_path.write_text(json.dumps({
        "velocity": {"status": "gold", "confidence": 0.81}
    }))
    reg = PatternRegistry.load(registry_path, results)
    assert reg.get("velocity").status == "gold"
```

---

## Deliverables checklist

Claude Code must confirm all of the following before the phase is complete:

- [ ] `uv run pytest tests/ -q` passes with zero failures
- [ ] `uv run python data/generate_sample.py` creates `data/sample_transactions.csv`
- [ ] `uv run python -m src.engine.experiment --pattern velocity --dataset data/sample_transactions.csv` runs end-to-end and prints a result summary
- [ ] `uv run python -m src.engine.experiment --pattern logistic --dataset data/sample_transactions.csv` runs end-to-end
- [ ] `results.tsv` exists and has 2 data rows (one per experiment)
- [ ] `registry.json` exists and reflects both patterns
- [ ] `scoring_policy.yaml` is unchanged from the spec (verify weights sum to 1.0)
- [ ] `CLAUDE.md` is present and unmodified

## What Claude Code should NOT do in Phase 1

- Do not build the dashboard, API, or any web server
- Do not implement the QWEN planner
- Do not implement the pattern arena
- Do not add any dependency not listed in pyproject.toml
- Do not modify `scoring_policy.yaml`
- Do not add a CLI framework — argparse only

---

*End of Phase 1 spec. Next: Phase 2 — Arena + QWEN planner.*
