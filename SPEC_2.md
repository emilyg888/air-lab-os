# SPEC.md — Phase 1: Foundation

# fraud-engine, built on bb_datasets

> **For Claude Code.** Read this file in full before writing any code.
> Build everything in this spec, in the order listed. Do not proceed to
> the next section until the current section's deliverables pass.
>
> **Important context:** `bb_datasets/` is an existing repo with working
> fraud detectors, feature engineering, and a DuckDB-backed data store.
> The fraud-engine wraps and governs that work — it does not replace it.
> Read `bb_datasets/fraud/detector.py`, `features.py`, and `load.py`
> before writing any code.

---

## What already exists in bb_datasets

This is the baseline you are building on. Do not rewrite it.

### Data schema (three tables in DuckDB)

**transactions** — the primary table for fraud detection

```
txn_id            str        e.g. "T00001"
account_id        str        e.g. "A042"
amount            float      AUD, signed (negative = debit)
txn_type          str        "CREDIT" | "DEBIT"
merchant_category str        "Sales" | "Payroll" | "Utilities" | "Supplier" | "Unknown"
timestamp         str        ISO 8601 datetime
fraud_flag        bool       ground truth label (set by generator_v3)
```

**accounts** (joined in by `load_transactions()`)

```
account_id    str
customer_id   str
account_type  str     "Business"
balance       float   AUD, log-normal
currency      str     "AUD"
```

**customers** (joined in by `load_transactions()`)

```
customer_id   str
risk_rating   str     "Low" | "Medium" | "High"
industry      str     "Retail" | "Tech" | "Manufacturing"
```

`load_transactions()` in `bb_datasets/fraud/load.py` returns a single
joined DataFrame with all of the above columns available.

### Engineered features (already built in bb_datasets/fraud/features.py)

`build_features(df)` adds these columns — use them, don't recompute them:

```
abs_amount       float   df["amount"].abs()
txn_count        int     per-account total transaction count
same_ts_count    int     txns sharing (account_id, timestamp) — catches velocity bursts
z_score          float   global z-score on abs_amount
account_zscore   float   per-account z-score on abs_amount
is_burst         bool    same_ts_count >= 3
account_had_burst bool   any txn from this account is a burst account
```

### Existing detectors (in bb_datasets/fraud/detector.py)

These are the patterns the registry will govern. Study their interfaces:

| Detector                 | Key signals                       | Notes                               |
| ------------------------ | --------------------------------- | ----------------------------------- |
| `detect_fraud` (v1)      | spike + velocity + global z       | Rule-based, configurable thresholds |
| `detect_fraud_v2`        | spike + velocity + account_zscore | Per-account z replaces global       |
| `detect_fraud_v3`        | weighted ensemble (0.4/0.4/0.2)   | Returns `fraud_score` float         |
| `detect_fraud_v4`        | v1 + account_had_burst            | Catches base txn, hurts precision   |
| `detect_fraud_v5`        | v1 + temporal burst (same day)    | Localised, better precision than v4 |
| `detect_fraud_ml`        | Logistic Regression (full fit)    | Training F1, not generalisation     |
| `detect_fraud_ml_honest` | Logistic Regression (70/30 split) | Proper held-out eval                |

All detectors consume `df` (the featured DataFrame) and return `df` with
`predicted_fraud` (bool) and optionally `fraud_score` (float) added.

### Three injected fraud patterns in the data

Pattern 1 — **Large spike**: 1% of txns, amount × 50. `fraud_flag=True`.
Pattern 2 — **Velocity burst**: 2% of txns emit 5 same-amount DEBITs at
`base_time`. `fraud_flag=True`. Caught by `same_ts_count`.
Pattern 3 — **High-risk amplification**: High-risk customers get 1.5×–3×
amounts. NOT labelled fraud — a behavioural amplifier, not an anomaly.

---

## Repo layout

Create this directory structure alongside `bb_datasets/`:

```
fraud-engine/
  CLAUDE.md                    ← standing instructions (content below)
  SPEC.md                      ← this file
  scoring_policy.yaml          ← locked scoring weights (content below)
  results.tsv                  ← experiment log (append-only)
  registry.json                ← derived state (rebuilt on startup)
  pyproject.toml               ← dependencies
  src/
    __init__.py
    registry.py                ← PatternRegistry class
    detectors/
      __init__.py
      base.py                  ← Detector ABC + DetectionResult
      rule_v1.py               ← wraps detect_fraud (v1)
      rule_v2.py               ← wraps detect_fraud_v2
      rule_v3.py               ← wraps detect_fraud_v3
      ml_logistic.py           ← wraps detect_fraud_ml_honest
    engine/
      __init__.py
      experiment.py            ← run_experiment()
      evaluate.py              ← evaluate() + load_policy()
      data.py                  ← load_bb_dataset() — loads from DuckDB + builds features
  tests/
    test_detectors.py
    test_experiment.py
    test_evaluate.py
    test_registry.py
```

The `bb_datasets/` repo sits alongside `fraud-engine/` and is imported
as a sibling package. Do not copy code from it — import it directly.

---

## CLAUDE.md

Write this content exactly to `CLAUDE.md`:

```markdown
# CLAUDE.md — fraud-engine

Standing instructions for all Claude Code sessions in this repo.
Read this before reading SPEC.md or any other file.

## What this repo is

An autonomous fraud detection research system built on top of bb_datasets.
An experiment loop runs detectors against SME banking transaction data,
scores them with a locked evaluation policy, and tracks results in a
registry. Detectors live in bb_datasets/fraud/detector.py — this repo
wraps and governs them, it does not replace them.

## Repo relationship
```

~/LocalDocuments/
bb_datasets/ ← existing data + detector code (read, don't modify)
fraud/
detector.py ← all detector functions
features.py ← build_features()
load.py ← load_transactions() from DuckDB
datasets/
generator_v3.py ← generate_fraud_dataset()
exports/
duckdb/
sandbox.db ← the live database
fraud-engine/ ← this repo (you build here)
src/
tests/

````

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
   `run_experiment()` always evaluates on the same data slice:
   the last 20% of rows sorted by txn_id. Never change this split logic.

5. **Run tests before every commit.**
   `uv run pytest tests/ -q` must pass with zero failures.

6. **One commit per experiment.**
   Kept result: `git commit -m "exp: <pattern_name> score=<score>"`
   Discarded:   `git reset --soft HEAD~1`

7. **Never install packages not in pyproject.toml.**
   If you need a new dependency, stop and ask the human.

8. **Never modify bb_datasets/ code.**
   If a detector in bb_datasets has a bug, surface it to the human.
   Do not patch it silently.

## File ownership

| File | Owner | Claude Code may… |
|------|-------|-----------------|
| scoring_policy.yaml | Human | Read only |
| CLAUDE.md | Human | Read only |
| SPEC.md | Human | Read only |
| bb_datasets/** | bb_datasets | Read only |
| results.tsv | System | Append only |
| registry.json | System | Overwrite via PatternRegistry.save() |
| src/** | Claude Code | Read + write |
| tests/** | Claude Code | Read + write |

## How to run

```bash
# Run one experiment
uv run python -m src.engine.experiment --pattern rule_v1 \
  --db ../bb_datasets/exports/duckdb/sandbox.db

# Run all tests
uv run pytest tests/ -q
````

## Session startup checklist

1. Read CLAUDE.md (this file)
2. Read SPEC.md
3. Read bb_datasets/fraud/detector.py, features.py, load.py
4. Run `uv run pytest tests/ -q` to confirm baseline
5. Check `results.tsv` for experiment history
6. Check `registry.json` for current registry state

````

---

## scoring_policy.yaml

Write this content exactly. Do not modify during the build:

```yaml
# Fraud detection scoring policy
# Locked — do not modify. Human-owned.
# Weights must sum to 1.0.

version: "1.0"

weights:
  precision_recall: 0.40   # F1 score against fraud_flag ground truth
  explainability:   0.25   # fraction of flagged txns with non-empty explanation
  latency:          0.20   # inverse-scaled: 0ms=1.0, 500ms=0.0, linear
  cost:             0.15   # inverse-scaled: $0=1.0, $0.10/1k=0.0, linear

promotion:
  silver_threshold: 0.65   # minimum score to achieve silver status
  gold_threshold:   0.78   # minimum score to become promotion candidate
  min_runs:         3      # minimum experiment runs before promotion eligible

latency:
  max_ms: 500              # rule detectors are fast; 500ms is generous ceiling

cost:
  max_per_1k: 0.10         # cost at which cost_score = 0.0 (local models = 0)
````

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
    "duckdb>=0.10",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
pythonpath = ["..", "."]
```

Note: `duckdb` is required because `src/engine/data.py` calls
`load_transactions()` from bb_datasets, which uses DuckDB.

---

## Component 1: Data loader

### `src/engine/data.py`

This module is the bridge between fraud-engine and bb_datasets.
It imports directly from the sibling repo — no copying.

```python
"""
Data loader — bridge to bb_datasets.

Loads the full joined transaction DataFrame from DuckDB,
builds features, and returns train/eval splits.

The eval split is FIXED: last 20% of rows sorted by txn_id.
Never change this — it makes experiments comparable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Resolve bb_datasets as a sibling of fraud-engine
_BB = Path(__file__).resolve().parent.parent.parent / "bb_datasets"
if str(_BB) not in sys.path:
    sys.path.insert(0, str(_BB))

from fraud.load import load_transactions          # noqa: E402
from fraud.features import build_features         # noqa: E402


DEFAULT_DB = _BB / "exports" / "duckdb" / "sandbox.db"


def load_bb_dataset(
    db_path: str | Path = DEFAULT_DB,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load, feature-engineer, and split the bb_datasets transaction data.

    Returns:
        train_df: first 80% of rows sorted by txn_id, with features
        eval_df:  last  20% of rows sorted by txn_id, with features

    The split is deterministic — same rows every time regardless of
    when the database was last regenerated.
    """
    df = load_transactions(db_path=str(db_path))
    df = build_features(df)

    # Sort by txn_id for deterministic, reproducible split
    df = df.sort_values("txn_id").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    eval_df  = df.iloc[split_idx:].copy()
    return train_df, eval_df


def dataset_summary(db_path: str | Path = DEFAULT_DB) -> dict:
    """Quick summary of the loaded dataset for logging."""
    df = load_transactions(db_path=str(db_path))
    df = build_features(df)
    total = len(df)
    fraud = int(df["fraud_flag"].sum())
    return {
        "total_rows": total,
        "fraud_rows": fraud,
        "fraud_rate": round(fraud / total, 4) if total else 0.0,
        "eval_rows":  total - int(total * 0.8),
    }
```

---

## Component 2: Detector interface

### `src/detectors/base.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
import pandas as pd


@dataclass
class DetectionResult:
    """
    Standardised output for every detector.

    All three lists must be the same length as the input DataFrame.

    flags       — True if the transaction is flagged as fraud
    scores      — float in [0.0, 1.0]; confidence or fraud probability
    explanation — human-readable reason per flag ("" if unflagged)
    latency_ms  — wall-clock ms for the full detect() call (set by base)
    cost_per_1k — estimated USD per 1000 transactions (0.0 for local)
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
        assert all(0.0 <= s <= 1.0 for s in self.scores), \
            "scores must be in [0, 1]"


class Detector(ABC):
    """
    Abstract base for all fraud detectors.

    Subclasses implement run(). The base class wraps run() to record
    wall-clock latency automatically.

    Usage:
        detector = RuleV1Detector(spike_threshold=20000)
        result   = detector.detect(eval_df)
    """

    name: str = "base"

    def detect(self, data: pd.DataFrame) -> DetectionResult:
        """
        Public entry point. Times the call and injects latency_ms.
        Do NOT override — override run() instead.
        """
        t0 = time.perf_counter()
        result = self.run(data)
        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    @abstractmethod
    def run(self, data: pd.DataFrame) -> DetectionResult:
        """
        Implement fraud detection logic here.

        Args:
            data: Featured DataFrame from load_bb_dataset().
                  Guaranteed columns: txn_id, account_id, amount,
                  txn_type, merchant_category, timestamp, fraud_flag,
                  abs_amount, same_ts_count, z_score, account_zscore,
                  account_had_burst, risk_rating, industry, balance.

        Returns:
            DetectionResult — do NOT set latency_ms, the base class does.
        """
        ...

    def describe(self) -> dict:
        """Return hyperparameter dict for TSV logging. Override to extend."""
        return {"detector": self.name}
```

### `src/detectors/rule_v1.py`

Wraps `detect_fraud` from bb_datasets. Adds per-flag explanation strings
(the existing detector does not produce them — we generate them here).

```python
"""
RuleV1Detector — wraps bb_datasets detect_fraud (v1).

Three OR-rules: spike | velocity | anomaly.
Explanation string is generated from which rules fired.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_BB = Path(__file__).resolve().parent.parent.parent.parent / "bb_datasets"
if str(_BB) not in sys.path:
    sys.path.insert(0, str(_BB))

from fraud.detector import detect_fraud, DEFAULTS  # noqa: E402
from .base import Detector, DetectionResult


class RuleV1Detector(Detector):
    """
    Rule-based detector v1: spike + velocity + global z-score.

    Args:
        spike_threshold:   abs_amount above which a txn is a spike
        same_ts_threshold: same-timestamp txn count to trigger velocity flag
        zscore_threshold:  global z-score magnitude to trigger anomaly flag
    """

    name = "rule_v1"

    def __init__(
        self,
        spike_threshold:   float = DEFAULTS["spike_threshold"],
        same_ts_threshold: int   = DEFAULTS["same_ts_threshold"],
        zscore_threshold:  float = DEFAULTS["zscore_threshold"],
    ):
        self.spike_threshold   = spike_threshold
        self.same_ts_threshold = same_ts_threshold
        self.zscore_threshold  = zscore_threshold

    def run(self, data: pd.DataFrame) -> DetectionResult:
        scored = detect_fraud(
            data,
            spike_threshold=self.spike_threshold,
            same_ts_threshold=self.same_ts_threshold,
            zscore_threshold=self.zscore_threshold,
        )

        flags        = scored["predicted_fraud"].tolist()
        scores       = []
        explanations = []

        for _, row in scored.iterrows():
            reasons = []
            if row["is_spike"]:
                reasons.append(
                    f"spike: abs_amount={row['abs_amount']:.0f} "
                    f"> threshold={self.spike_threshold:.0f}"
                )
            if row["is_velocity"]:
                reasons.append(
                    f"velocity: {row['same_ts_count']} txns at same timestamp "
                    f"(threshold={self.same_ts_threshold})"
                )
            if row["is_anomaly"]:
                reasons.append(
                    f"anomaly: z_score={row['z_score']:.2f} "
                    f"> threshold={self.zscore_threshold}"
                )

            if reasons:
                conf = 0.95 if (row["is_spike"] or row["is_velocity"]) else 0.65
                scores.append(conf)
                explanations.append("; ".join(reasons))
            else:
                scores.append(0.0)
                explanations.append("")

        return DetectionResult(
            flags=flags,
            scores=scores,
            explanation=explanations,
            cost_per_1k=0.0,
        )

    def describe(self) -> dict:
        return {
            "detector":          self.name,
            "spike_threshold":   self.spike_threshold,
            "same_ts_threshold": self.same_ts_threshold,
            "zscore_threshold":  self.zscore_threshold,
        }
```

### `src/detectors/rule_v2.py`

```python
"""
RuleV2Detector — wraps bb_datasets detect_fraud_v2.

Uses per-account z-score instead of global — catches txns that are
anomalous for a specific account even if globally unremarkable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_BB = Path(__file__).resolve().parent.parent.parent.parent / "bb_datasets"
if str(_BB) not in sys.path:
    sys.path.insert(0, str(_BB))

from fraud.detector import detect_fraud_v2  # noqa: E402
from .base import Detector, DetectionResult


class RuleV2Detector(Detector):
    """
    Rule-based detector v2: spike + velocity + per-account z-score.

    Args:
        spike_threshold:          abs_amount spike threshold
        same_ts_threshold:        velocity burst threshold
        account_zscore_threshold: per-account z-score threshold
    """

    name = "rule_v2"

    def __init__(
        self,
        spike_threshold:          float = 15_000,
        same_ts_threshold:        int   = 3,
        account_zscore_threshold: float = 2.5,
    ):
        self.spike_threshold          = spike_threshold
        self.same_ts_threshold        = same_ts_threshold
        self.account_zscore_threshold = account_zscore_threshold

    def run(self, data: pd.DataFrame) -> DetectionResult:
        scored = detect_fraud_v2(
            data,
            spike_threshold=self.spike_threshold,
            same_ts_threshold=self.same_ts_threshold,
            account_zscore_threshold=self.account_zscore_threshold,
        )

        flags        = scored["predicted_fraud"].tolist()
        scores       = []
        explanations = []

        for _, row in scored.iterrows():
            reasons = []
            if row["is_spike"]:
                reasons.append(
                    f"spike: abs_amount={row['abs_amount']:.0f} "
                    f"> {self.spike_threshold:.0f}"
                )
            if row["is_velocity"]:
                reasons.append(
                    f"velocity: {row['same_ts_count']} same-ts txns"
                )
            if row["is_anomaly"]:
                reasons.append(
                    f"account_zscore={row['account_zscore']:.2f} "
                    f"> {self.account_zscore_threshold}"
                )
            if reasons:
                conf = 0.95 if (row["is_spike"] or row["is_velocity"]) else 0.70
                scores.append(conf)
                explanations.append("; ".join(reasons))
            else:
                scores.append(0.0)
                explanations.append("")

        return DetectionResult(
            flags=flags,
            scores=scores,
            explanation=explanations,
            cost_per_1k=0.0,
        )

    def describe(self) -> dict:
        return {
            "detector":                self.name,
            "spike_threshold":         self.spike_threshold,
            "same_ts_threshold":       self.same_ts_threshold,
            "account_zscore_threshold": self.account_zscore_threshold,
        }
```

### `src/detectors/rule_v3.py`

```python
"""
RuleV3Detector — wraps bb_datasets detect_fraud_v3.

Weighted ensemble: fraud_score = spike*0.4 + velocity*0.4 + anomaly*0.2.
Exposes score_threshold as a tunable hyperparameter.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_BB = Path(__file__).resolve().parent.parent.parent.parent / "bb_datasets"
if str(_BB) not in sys.path:
    sys.path.insert(0, str(_BB))

from fraud.detector import detect_fraud_v3  # noqa: E402
from .base import Detector, DetectionResult


class RuleV3Detector(Detector):
    """
    Weighted ensemble detector v3.

    Args:
        score_threshold: fraud_score cutoff for flagging (default 0.5)
                         > 0.4 requires at least 2 signals at default weights
                         > 0.3 allows any spike or velocity alone to fire
    """

    name = "rule_v3"

    def __init__(self, score_threshold: float = 0.5):
        self.score_threshold = score_threshold

    def run(self, data: pd.DataFrame) -> DetectionResult:
        scored = detect_fraud_v3(data)

        predicted = (scored["fraud_score"] > self.score_threshold).tolist()
        scores    = scored["fraud_score"].clip(0.0, 1.0).tolist()

        explanations = []
        for _, row in scored.iterrows():
            if row["fraud_score"] > self.score_threshold:
                parts = []
                if row["is_spike"]:
                    parts.append(f"spike(w=0.4, amt={row['abs_amount']:.0f})")
                if row["is_velocity"]:
                    parts.append(f"velocity(w=0.4, count={row['same_ts_count']})")
                if row["is_anomaly"]:
                    parts.append(f"anomaly(w=0.2, z={row['z_score']:.2f})")
                explanations.append(
                    f"score={row['fraud_score']:.2f}: " + "; ".join(parts)
                )
            else:
                explanations.append("")

        return DetectionResult(
            flags=predicted,
            scores=[float(s) for s in scores],
            explanation=explanations,
            cost_per_1k=0.0,
        )

    def describe(self) -> dict:
        return {"detector": self.name, "score_threshold": self.score_threshold}
```

### `src/detectors/ml_logistic.py`

```python
"""
MLLogisticDetector — wraps bb_datasets detect_fraud_ml_honest.

Uses a proper 70/30 stratified train/test split. Only eval-split rows
get predictions; the rest are treated as TN (predicted_fraud=False).
Explanation: top contributing feature per prediction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_BB = Path(__file__).resolve().parent.parent.parent.parent / "bb_datasets"
if str(_BB) not in sys.path:
    sys.path.insert(0, str(_BB))

from fraud.detector import (  # noqa: E402
    detect_fraud_ml_honest,
    _ml_honest_model,
    _ML_FEATURES,
)
from .base import Detector, DetectionResult


class MLLogisticDetector(Detector):
    """
    Logistic Regression detector with honest train/test split.

    The underlying model is a singleton in bb_datasets/detector.py —
    trained once per process. This wrapper adds explanation strings.
    """

    name = "ml_logistic"

    def __init__(self, score_threshold: float = 0.5):
        self.score_threshold = score_threshold

    def run(self, data: pd.DataFrame) -> DetectionResult:
        scored = detect_fraud_ml_honest(data)

        predicted    = scored["predicted_fraud"].tolist()
        fraud_scores = scored["fraud_score"].fillna(0.0).clip(0.0, 1.0).tolist()
        explanations = self._build_explanations(scored)

        return DetectionResult(
            flags=predicted,
            scores=[float(s) for s in fraud_scores],
            explanation=explanations,
            cost_per_1k=0.0,
        )

    def _build_explanations(self, scored: pd.DataFrame) -> list[str]:
        """Top contributing feature for each flagged prediction."""
        global _ml_honest_model

        explanations = []
        if _ml_honest_model is None:
            return ["" for _ in range(len(scored))]

        coefs = _ml_honest_model.coef_[0]

        for _, row in scored.iterrows():
            if np.isnan(row.get("fraud_score", float("nan"))):
                explanations.append("")
                continue
            if not row["predicted_fraud"]:
                explanations.append("")
                continue

            feat_vals = [float(row.get(col, 0.0)) for col in _ML_FEATURES]
            contributions = coefs * np.array(feat_vals)
            top_idx   = int(np.argmax(np.abs(contributions)))
            top_feat  = _ML_FEATURES[top_idx]
            direction = "high" if contributions[top_idx] > 0 else "low"
            explanations.append(
                f"top feature: {top_feat} ({direction}, "
                f"coef={coefs[top_idx]:.3f}, val={feat_vals[top_idx]:.3f})"
            )

        return explanations

    def describe(self) -> dict:
        return {"detector": self.name, "score_threshold": self.score_threshold}
```

---

## Component 3: Evaluation engine

### `src/engine/evaluate.py`

```python
"""
Evaluation engine — locked scoring layer.

evaluate(result, labels) → EvaluationMetrics

Reads weights from scoring_policy.yaml. Do not hardcode weights here.
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
    explainability_weight:   float
    latency_weight:          float
    cost_weight:             float
    latency_max_ms:          float
    cost_max_per_1k:         float
    silver_threshold:        float
    gold_threshold:          float
    min_runs:                int

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
    """Load and validate the scoring policy. Raises on bad weights."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    w = raw["weights"]
    policy = ScoringPolicy(
        precision_recall_weight = w["precision_recall"],
        explainability_weight   = w["explainability"],
        latency_weight          = w["latency"],
        cost_weight             = w["cost"],
        latency_max_ms          = raw["latency"]["max_ms"],
        cost_max_per_1k         = raw["cost"]["max_per_1k"],
        silver_threshold        = raw["promotion"]["silver_threshold"],
        gold_threshold          = raw["promotion"]["gold_threshold"],
        min_runs                = raw["promotion"]["min_runs"],
    )
    policy.validate()
    return policy


@dataclass
class EvaluationMetrics:
    """Full breakdown. score is the weighted composite [0, 1]."""
    precision:              float
    recall:                 float
    f1:                     float
    precision_recall_score: float
    explainability_score:   float
    latency_score:          float
    cost_score:             float
    score:                  float


def evaluate(
    result: DetectionResult,
    labels: list[bool],
    policy: ScoringPolicy | None = None,
) -> EvaluationMetrics:
    """
    Score a DetectionResult against ground truth fraud_flag labels.

    Args:
        result: output of Detector.detect()
        labels: fraud_flag column from the eval DataFrame, same length
        policy: loaded ScoringPolicy; loads from yaml if None
    """
    if policy is None:
        policy = load_policy()

    assert len(result.flags) == len(labels), (
        f"result has {len(result.flags)} flags but labels has {len(labels)}"
    )

    y_true = [int(l) for l in labels]
    y_pred = [int(f) for f in result.flags]

    if sum(y_pred) == 0:
        precision = recall = f1 = 0.0
    else:
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall    = float(recall_score(y_true, y_pred, zero_division=0))
        f1        = float(f1_score(y_true, y_pred, zero_division=0))

    pr_score = f1

    flagged = [i for i, f in enumerate(result.flags) if f]
    if flagged:
        explained  = sum(1 for i in flagged if result.explanation[i].strip())
        expl_score = explained / len(flagged)
    else:
        expl_score = 1.0

    latency_score = max(0.0, 1.0 - result.latency_ms / policy.latency_max_ms)
    cost_score    = max(0.0, 1.0 - result.cost_per_1k / policy.cost_max_per_1k)

    composite = (
        policy.precision_recall_weight * pr_score
        + policy.explainability_weight * expl_score
        + policy.latency_weight        * latency_score
        + policy.cost_weight           * cost_score
    )

    return EvaluationMetrics(
        precision              = round(precision, 4),
        recall                 = round(recall, 4),
        f1                     = round(f1, 4),
        precision_recall_score = round(pr_score, 4),
        explainability_score   = round(expl_score, 4),
        latency_score          = round(latency_score, 4),
        cost_score             = round(cost_score, 4),
        score                  = round(composite, 4),
    )
```

---

## Component 4: Experiment engine

### `src/engine/experiment.py`

```python
"""
Experiment engine — run_experiment() is the core loop entry point.

The eval window is always the last 20% of rows sorted by txn_id.
This is fixed — never change it.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from src.detectors.base import Detector, DetectionResult
from src.detectors.rule_v1 import RuleV1Detector
from src.detectors.rule_v2 import RuleV2Detector
from src.detectors.rule_v3 import RuleV3Detector
from src.detectors.ml_logistic import MLLogisticDetector
from src.engine.data import load_bb_dataset, DEFAULT_DB
from src.engine.evaluate import evaluate, load_policy, EvaluationMetrics
from src.registry import PatternRegistry


PATTERN_MAP: dict[str, type[Detector]] = {
    "rule_v1":     RuleV1Detector,
    "rule_v2":     RuleV2Detector,
    "rule_v3":     RuleV3Detector,
    "ml_logistic": MLLogisticDetector,
}


@dataclass
class ExperimentResult:
    pattern:     str
    db_path:     str
    config:      dict
    metrics:     EvaluationMetrics | None
    score:       float
    commit:      str
    status:      str    # "keep" | "discard" | "crash"
    description: str
    timestamp:   int


def _get_short_commit() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return r.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no-git"


def run_experiment(
    pattern:       str,
    db_path:       str | Path = DEFAULT_DB,
    config:        dict | None = None,
    description:   str = "",
    registry_path: Path = Path("registry.json"),
    results_path:  Path = Path("results.tsv"),
) -> ExperimentResult:
    """
    Run one experiment end-to-end against the bb_datasets DuckDB.

    Side effects: appends to results.tsv, rebuilds registry.json.
    """
    config    = config or {}
    commit    = _get_short_commit()
    timestamp = int(time.time())

    if pattern not in PATTERN_MAP:
        exp = ExperimentResult(
            pattern=pattern, db_path=str(db_path), config=config,
            metrics=None, score=0.0, commit=commit, status="crash",
            description=(
                f"CRASH: unknown pattern '{pattern}'. "
                f"Available: {list(PATTERN_MAP.keys())}"
            ),
            timestamp=timestamp,
        )
        _append_tsv(exp, results_path)
        _rebuild_registry(registry_path, results_path)
        return exp

    try:
        detector = PATTERN_MAP[pattern](**config)
        _, eval_df = load_bb_dataset(db_path)
        result: DetectionResult = detector.detect(eval_df)

        policy  = load_policy()
        labels  = eval_df["fraud_flag"].astype(bool).tolist()
        metrics = evaluate(result, labels, policy)
        score   = metrics.score

        registry     = PatternRegistry.load(registry_path, results_path)
        current_best = registry.best_score(pattern)
        status = "keep" if (current_best is None or score > current_best) else "discard"

        exp = ExperimentResult(
            pattern=pattern, db_path=str(db_path), config=config,
            metrics=metrics, score=score, commit=commit, status=status,
            description=description or f"{pattern} config={config}",
            timestamp=timestamp,
        )

    except Exception as e:
        exp = ExperimentResult(
            pattern=pattern, db_path=str(db_path), config=config,
            metrics=None, score=0.0, commit=commit, status="crash",
            description=f"CRASH: {type(e).__name__}: {e}",
            timestamp=timestamp,
        )

    _append_tsv(exp, results_path)
    _rebuild_registry(registry_path, results_path)
    return exp


def _append_tsv(result: ExperimentResult, path: Path) -> None:
    header = (
        "commit\tpattern\tscore\tprecision\trecall\tf1\t"
        "expl_score\tlatency_score\tcost_score\t"
        "status\tdescription\ttimestamp\n"
    )
    if not path.exists():
        path.write_text(header)

    m = result.metrics
    if m is not None:
        row = (
            f"{result.commit}\t{result.pattern}\t{result.score:.6f}\t"
            f"{m.precision:.4f}\t{m.recall:.4f}\t{m.f1:.4f}\t"
            f"{m.explainability_score:.4f}\t"
            f"{m.latency_score:.4f}\t{m.cost_score:.4f}\t"
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


def _rebuild_registry(registry_path: Path, results_path: Path) -> None:
    registry = PatternRegistry.load(registry_path, results_path)
    registry.save(registry_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run one fraud detection experiment")
    parser.add_argument("--pattern", required=True, choices=list(PATTERN_MAP.keys()))
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Path to sandbox.db")
    parser.add_argument("--config", default="{}", help="JSON hyperparameter overrides")
    parser.add_argument("--description", default="")
    args = parser.parse_args()

    result = run_experiment(
        pattern=args.pattern,
        db_path=args.db,
        config=json.loads(args.config),
        description=args.description,
    )

    print(f"\n--- Experiment result ---")
    print(f"pattern:     {result.pattern}")
    print(f"status:      {result.status}")
    print(f"score:       {result.score:.6f}")
    if result.metrics:
        m = result.metrics
        print(f"f1:             {m.f1:.4f}")
        print(f"precision:      {m.precision:.4f}")
        print(f"recall:         {m.recall:.4f}")
        print(f"explainability: {m.explainability_score:.4f}")
        print(f"latency_score:  {m.latency_score:.4f}")
    print(f"commit:      {result.commit}")
    print(f"description: {result.description}")
```

---

## Component 5: Pattern registry

### `src/registry.py`

```python
"""
Pattern registry — live system state rebuilt from results.tsv.

Never write registry.json directly.
Only call PatternRegistry.save() after loading from the TSV.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class RegistryEntry:
    pattern:             str
    status:              str    # bronze | silver | gold | discard | crash
    confidence:          float  # best score across all "keep" runs
    runs:                int    # total experiment attempts
    last_commit:         str
    last_updated:        int    # unix timestamp
    promotion_candidate: bool
    best_precision:      float = 0.0
    best_recall:         float = 0.0
    best_f1:             float = 0.0
    description:         str   = ""


class PatternRegistry:
    """Manages pattern state. Rebuilt from results.tsv on every load."""

    SILVER_THRESHOLD: float = 0.65
    GOLD_THRESHOLD:   float = 0.78
    MIN_RUNS:         int   = 3

    def __init__(self, entries: dict[str, RegistryEntry]):
        self._entries = entries

    @classmethod
    def load(cls, registry_path: Path, results_path: Path) -> "PatternRegistry":
        """Rebuild from results.tsv. Gold statuses from registry.json preserved."""
        try:
            from src.engine.evaluate import load_policy
            policy = load_policy()
            cls.SILVER_THRESHOLD = policy.silver_threshold
            cls.GOLD_THRESHOLD   = policy.gold_threshold
            cls.MIN_RUNS         = policy.min_runs
        except Exception:
            pass

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
                score   = float(row.get("score", 0))
                status  = row.get("status", "discard")
                ts      = int(row.get("timestamp", 0))

                if pattern not in entries:
                    entries[pattern] = RegistryEntry(
                        pattern=pattern, status="bronze", confidence=0.0,
                        runs=0, last_commit=row.get("commit", ""),
                        last_updated=ts, promotion_candidate=False,
                        description=row.get("description", ""),
                    )

                e = entries[pattern]
                e.runs        += 1
                e.last_updated = max(e.last_updated, ts)
                e.last_commit  = row.get("commit", e.last_commit)

                if status == "keep" and score > e.confidence:
                    e.confidence     = score
                    e.best_precision = float(row.get("precision", 0))
                    e.best_recall    = float(row.get("recall", 0))
                    e.best_f1        = float(row.get("f1", 0))

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
        data = {p: asdict(e) for p, e in self._entries.items()}
        path.write_text(json.dumps(data, indent=2))
```

---

## Tests

### `tests/test_detectors.py`

```python
"""
Detector tests. Uses a synthetic featured DataFrame — no DuckDB required.
"""

import pytest
import pandas as pd
import numpy as np
from src.detectors.base import DetectionResult
from src.detectors.rule_v1 import RuleV1Detector
from src.detectors.rule_v2 import RuleV2Detector
from src.detectors.rule_v3 import RuleV3Detector


@pytest.fixture
def featured_df():
    """Minimal DataFrame matching bb_datasets build_features output."""
    n = 10
    return pd.DataFrame({
        "txn_id":            [f"T{i:05d}" for i in range(n)],
        "account_id":        ["A001"] * 6 + ["A002"] * 4,
        "amount":            [100.0, -200.0, 50.0, -300.0, 25000.0, -50.0,
                              80.0, -90.0, 110.0, -120.0],
        "txn_type":          ["CREDIT", "DEBIT"] * 5,
        "merchant_category": ["Sales", "Payroll", "Utilities", "Supplier", "Unknown",
                              "Sales", "Payroll", "Utilities", "Supplier", "Unknown"],
        "timestamp":         ["2024-01-01T10:00:00"] * 5
                             + ["2024-01-01T10:00:00"] * 3
                             + ["2024-01-02T10:00:00"] * 2,
        "fraud_flag":        [False, False, False, False, True,
                              False, True, True, True, False],
        "abs_amount":        [100.0, 200.0, 50.0, 300.0, 25000.0, 50.0,
                              80.0, 90.0, 110.0, 120.0],
        "txn_count":         [6] * 6 + [4] * 4,
        "same_ts_count":     [5, 5, 5, 5, 5, 5, 3, 3, 3, 1],
        "z_score":           [0.1, 0.2, -0.1, 0.3, 4.5, -0.2,
                              0.0, 0.1, 0.1, 0.1],
        "account_zscore":    [0.1, 0.2, -0.1, 0.3, 3.5, -0.2,
                              0.0, 2.8, 0.1, 0.1],
        "is_burst":          [True] * 6 + [True, True, True, False],
        "account_had_burst": [True] * 6 + [True, True, True, False],
        "risk_rating":       ["Medium"] * 10,
        "industry":          ["Tech"] * 10,
        "balance":           [50000.0] * 10,
        "account_balance":   [50000.0] * 10,
        "account_type":      ["Business"] * 10,
    })


def test_rule_v1_result_shape(featured_df):
    r = RuleV1Detector().detect(featured_df)
    assert len(r.flags) == len(featured_df)
    assert len(r.scores) == len(featured_df)
    assert len(r.explanation) == len(featured_df)


def test_rule_v1_catches_spike(featured_df):
    r = RuleV1Detector(spike_threshold=10_000).detect(featured_df)
    assert r.flags[4] is True  # abs_amount=25000 > 10000


def test_rule_v1_catches_velocity(featured_df):
    r = RuleV1Detector(same_ts_threshold=3).detect(featured_df)
    assert any(r.flags[i] for i in range(6))  # same_ts_count=5 >= 3


def test_rule_v1_explanation_on_flag(featured_df):
    r = RuleV1Detector(spike_threshold=10_000).detect(featured_df)
    for i, flag in enumerate(r.flags):
        if flag:
            assert r.explanation[i] != "", f"Flagged row {i} missing explanation"


def test_rule_v1_scores_in_range(featured_df):
    r = RuleV1Detector().detect(featured_df)
    assert all(0.0 <= s <= 1.0 for s in r.scores)


def test_rule_v2_result_shape(featured_df):
    r = RuleV2Detector().detect(featured_df)
    assert len(r.flags) == len(featured_df)


def test_rule_v3_returns_float_scores(featured_df):
    r = RuleV3Detector().detect(featured_df)
    assert all(isinstance(s, float) for s in r.scores)


def test_latency_recorded(featured_df):
    r = RuleV1Detector().detect(featured_df)
    assert r.latency_ms > 0


def test_unflagged_rows_empty_explanation(featured_df):
    r = RuleV1Detector(spike_threshold=1_000_000).detect(featured_df)
    assert all(e == "" for e in r.explanation)
```

### `tests/test_evaluate.py`

```python
import pytest
from src.detectors.base import DetectionResult
from src.engine.evaluate import evaluate, load_policy


@pytest.fixture
def policy():
    return load_policy()


def _result(n_fraud=3, n_total=10, latency_ms=10.0, with_explanation=True):
    flags = [True] * n_fraud + [False] * (n_total - n_fraud)
    scores = [0.9] * n_fraud + [0.1] * (n_total - n_fraud)
    expl = (
        ["velocity burst: 5 same-ts txns"] * n_fraud + [""] * (n_total - n_fraud)
        if with_explanation else [""] * n_total
    )
    return DetectionResult(flags=flags, scores=scores, explanation=expl,
                           latency_ms=latency_ms)


def _labels(n_fraud=3, n_total=10):
    return [True] * n_fraud + [False] * (n_total - n_fraud)


def test_perfect_precision_recall(policy):
    m = evaluate(_result(), _labels(), policy)
    assert m.precision == pytest.approx(1.0)
    assert m.recall    == pytest.approx(1.0)
    assert m.f1        == pytest.approx(1.0)


def test_score_in_range(policy):
    m = evaluate(_result(), _labels(), policy)
    assert 0.0 <= m.score <= 1.0


def test_explainability_full(policy):
    m = evaluate(_result(with_explanation=True), _labels(), policy)
    assert m.explainability_score == pytest.approx(1.0)


def test_explainability_zero(policy):
    m = evaluate(_result(with_explanation=False), _labels(), policy)
    assert m.explainability_score == pytest.approx(0.0)


def test_no_flags_gives_zero_f1(policy):
    r = DetectionResult(flags=[False]*10, scores=[0.0]*10,
                        explanation=[""]*10, latency_ms=5.0)
    m = evaluate(r, _labels(), policy)
    assert m.f1 == 0.0


def test_latency_penalised(policy):
    fast = _result(latency_ms=0.0)
    slow = _result(latency_ms=500.0)
    labels = _labels()
    assert evaluate(fast, labels, policy).latency_score > \
           evaluate(slow, labels, policy).latency_score
    assert evaluate(slow, labels, policy).latency_score == pytest.approx(0.0)


def test_weights_sum_to_one(policy):
    total = (
        policy.precision_recall_weight + policy.explainability_weight
        + policy.latency_weight + policy.cost_weight
    )
    assert abs(total - 1.0) < 1e-6
```

### `tests/test_experiment.py`

```python
"""
Experiment engine tests. Mocks load_bb_dataset() — no DuckDB required.
"""

import json
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch

from src.engine.experiment import run_experiment, PATTERN_MAP


MOCK_DF = pd.DataFrame({
    "txn_id":            [f"T{i:05d}" for i in range(50)],
    "account_id":        [f"A{i%5:03d}" for i in range(50)],
    "amount":            [float(i * 100) for i in range(50)],
    "txn_type":          ["CREDIT" if i % 2 == 0 else "DEBIT" for i in range(50)],
    "merchant_category": ["Sales"] * 50,
    "timestamp":         ["2024-01-01T10:00:00"] * 50,
    "fraud_flag":        [i % 8 == 0 for i in range(50)],
    "abs_amount":        [float(i * 100) for i in range(50)],
    "txn_count":         [10] * 50,
    "same_ts_count":     [1] * 50,
    "z_score":           [0.0] * 50,
    "account_zscore":    [0.0] * 50,
    "is_burst":          [False] * 50,
    "account_had_burst": [False] * 50,
    "risk_rating":       ["Low"] * 50,
    "industry":          ["Tech"] * 50,
    "balance":           [10000.0] * 50,
    "account_balance":   [10000.0] * 50,
    "account_type":      ["Business"] * 50,
})


@pytest.fixture
def tmp_paths(tmp_path):
    return {"registry": tmp_path / "registry.json",
             "results":  tmp_path / "results.tsv"}


def mock_load(db_path=None):
    df = MOCK_DF.copy()
    split = int(len(df) * 0.8)
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def test_run_experiment_rule_v1(tmp_paths):
    with patch("src.engine.experiment.load_bb_dataset", side_effect=mock_load):
        r = run_experiment("rule_v1", registry_path=tmp_paths["registry"],
                           results_path=tmp_paths["results"])
    assert r.pattern == "rule_v1"
    assert r.status in ("keep", "discard", "crash")
    assert 0.0 <= r.score <= 1.0


def test_writes_tsv(tmp_paths):
    with patch("src.engine.experiment.load_bb_dataset", side_effect=mock_load):
        run_experiment("rule_v1", registry_path=tmp_paths["registry"],
                       results_path=tmp_paths["results"])
    lines = tmp_paths["results"].read_text().strip().split("\n")
    assert len(lines) == 2  # header + 1 data row


def test_updates_registry(tmp_paths):
    with patch("src.engine.experiment.load_bb_dataset", side_effect=mock_load):
        run_experiment("rule_v1", registry_path=tmp_paths["registry"],
                       results_path=tmp_paths["results"])
    data = json.loads(tmp_paths["registry"].read_text())
    assert "rule_v1" in data


def test_unknown_pattern_is_crash(tmp_paths):
    r = run_experiment("nonexistent", registry_path=tmp_paths["registry"],
                       results_path=tmp_paths["results"])
    assert r.status == "crash"


def test_all_patterns_runnable(tmp_paths):
    for pattern in PATTERN_MAP:
        with patch("src.engine.experiment.load_bb_dataset", side_effect=mock_load):
            r = run_experiment(pattern, registry_path=tmp_paths["registry"],
                               results_path=tmp_paths["results"])
        assert r.status in ("keep", "discard", "crash"), \
            f"{pattern} returned unexpected status: {r.status}"
```

### `tests/test_registry.py`

```python
import json
import pytest
from pathlib import Path
from src.registry import PatternRegistry


HEADER = (
    "commit\tpattern\tscore\tprecision\trecall\tf1\t"
    "expl_score\tlatency_score\tcost_score\t"
    "status\tdescription\ttimestamp\n"
)


def write_tsv(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        f.write(HEADER)
        for r in rows:
            f.write(
                f"{r['commit']}\t{r['pattern']}\t{r['score']:.6f}\t"
                f"{r.get('precision',0):.4f}\t{r.get('recall',0):.4f}\t"
                f"{r.get('f1',0):.4f}\t{r.get('expl_score',0):.4f}\t"
                f"0.9000\t1.0000\t"
                f"{r['status']}\t{r.get('description','')}\t"
                f"{r.get('timestamp',0)}\n"
            )


def test_loads_from_tsv(tmp_path):
    results = tmp_path / "results.tsv"
    registry = tmp_path / "registry.json"
    write_tsv(results, [{"commit":"abc1234","pattern":"rule_v1",
                          "score":0.72,"f1":0.72,"status":"keep","timestamp":1000}])
    reg = PatternRegistry.load(registry, results)
    e = reg.get("rule_v1")
    assert e is not None
    assert e.confidence == pytest.approx(0.72)
    assert e.runs == 1


def test_silver_status(tmp_path):
    results = tmp_path / "results.tsv"
    write_tsv(results, [{"commit":"abc","pattern":"rule_v1",
                          "score":0.70,"status":"keep","timestamp":1000}])
    reg = PatternRegistry.load(tmp_path/"r.json", results)
    assert reg.get("rule_v1").status == "silver"


def test_bronze_status(tmp_path):
    results = tmp_path / "results.tsv"
    write_tsv(results, [{"commit":"abc","pattern":"rule_v1",
                          "score":0.50,"status":"keep","timestamp":1000}])
    reg = PatternRegistry.load(tmp_path/"r.json", results)
    assert reg.get("rule_v1").status == "bronze"


def test_promotion_candidate(tmp_path):
    results = tmp_path / "results.tsv"
    rows = [{"commit":f"abc{i}","pattern":"rule_v1","score":0.82,
              "status":"keep","timestamp":i} for i in range(3)]
    write_tsv(results, rows)
    reg = PatternRegistry.load(tmp_path/"r.json", results)
    assert reg.get("rule_v1").promotion_candidate is True


def test_gold_preserved(tmp_path):
    results = tmp_path / "results.tsv"
    rpath   = tmp_path / "registry.json"
    write_tsv(results, [{"commit":"abc","pattern":"rule_v1",
                          "score":0.82,"status":"keep","timestamp":1000}])
    rpath.write_text(json.dumps({"rule_v1": {"status": "gold", "confidence": 0.82}}))
    reg = PatternRegistry.load(rpath, results)
    assert reg.get("rule_v1").status == "gold"


def test_discard_does_not_update_confidence(tmp_path):
    results = tmp_path / "results.tsv"
    write_tsv(results, [
        {"commit":"abc1","pattern":"rule_v1","score":0.72,"status":"keep","timestamp":1000},
        {"commit":"abc2","pattern":"rule_v1","score":0.60,"status":"discard","timestamp":2000},
    ])
    reg = PatternRegistry.load(tmp_path/"r.json", results)
    assert reg.get("rule_v1").confidence == pytest.approx(0.72)
    assert reg.get("rule_v1").runs == 2
```

---

## Deliverables checklist

Claude Code must confirm all of the following before Phase 1 is complete:

- [ ] `uv run pytest tests/ -q` passes with zero failures
- [ ] `uv run python -m src.engine.experiment --pattern rule_v1` prints a result summary
- [ ] `uv run python -m src.engine.experiment --pattern rule_v2` runs end-to-end
- [ ] `uv run python -m src.engine.experiment --pattern rule_v3` runs end-to-end
- [ ] `uv run python -m src.engine.experiment --pattern ml_logistic` runs end-to-end
- [ ] `results.tsv` has 4 data rows (one per pattern)
- [ ] `registry.json` reflects all 4 patterns with correct status (bronze/silver/gold)
- [ ] `scoring_policy.yaml` is unchanged (weights sum to 1.0)
- [ ] `CLAUDE.md` is present and unmodified
- [ ] No files in `bb_datasets/` were modified (verify with `git -C ../bb_datasets status`)

## What Claude Code should NOT do in Phase 1

- Do not rewrite or copy detector logic from bb_datasets — import it
- Do not build the dashboard, API, or web server
- Do not implement the QWEN planner
- Do not implement the pattern arena
- Do not add dependencies not in pyproject.toml
- Do not modify `scoring_policy.yaml`
- Do not modify anything in `bb_datasets/`
- Do not add a CLI framework — argparse only

---

_End of Phase 1 spec. Next: Phase 2 — Arena + QWEN planner._
