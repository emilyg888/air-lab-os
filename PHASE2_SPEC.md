# SPEC_2.md — air-lab-os: Phase 2 — Fraud Detection Plugin

> **For Claude Code.** Read CLAUDE.md, then SPEC.md, then this file.
> Phase 1 must be passing (`uv run pytest tests/ -q`) before starting here.
> This phase adds the fraud use case as a plugin. The engine (runtime/,
> lab/, evaluation/, memory/) is frozen — do not modify those files.
>
> All new code goes into: `use_cases/fraud/`
>
> Run the Phase 2 test suite at the end:
> `uv run pytest use_cases/fraud/tests/ -q`

---

## What Phase 2 delivers

Three things, all inside `use_cases/fraud/`:

1. **`FraudHandle`** — implements `DatasetHandle` against `bb_datasets`
2. **Three fraud patterns** — wrapping the existing `bb_datasets` detectors,
   each implementing `PatternHandler`
3. **A test suite** verifying the plugin contract is correctly satisfied

Nothing in the engine changes. Nothing in `bb_datasets` changes.

---

## Sibling repo layout reminder

```
~/LocalDocuments/
  air-lab-os/           ← this repo (engine + this plugin)
  bb_datasets/          ← read-only fraud data and detectors
```

`bb_datasets` path relative to this repo: `../bb_datasets/`

Before writing any code, read these files in `bb_datasets`:
- `fraud/load.py`       — `load_transactions()` returns joined DataFrame
- `fraud/features.py`   — `build_features()` adds engineered columns
- `fraud/detector.py`   — contains `detect_fraud` v1–v5, `detect_fraud_ml`,
                          `detect_fraud_ml_honest`

The DuckDB database lives at: `../bb_datasets/exports/duckdb/sandbox.db`

---

## Directory structure to create

```
air-lab-os/
  use_cases/
    __init__.py
    fraud/
      __init__.py
      handle.py              ← FraudHandle (DatasetHandle impl)
      patterns/
        __init__.py
        rule_spike.py        ← Pattern 1: large-amount spike rule
        rule_velocity.py     ← Pattern 2: same-timestamp burst rule
        ml_logistic.py       ← Pattern 3: logistic regression on features
      tests/
        __init__.py
        test_fraud_handle.py
        test_fraud_patterns.py
```

Patterns also need to be discoverable by the engine. Symlink or copy
each pattern file into `patterns/scratch/` so `main.py` can find them.
The canonical source lives in `use_cases/fraud/patterns/` — the engine
reads from `patterns/scratch/`.

---

## Component 1: FraudHandle

### `use_cases/fraud/handle.py`

```python
"""
FraudHandle — DatasetHandle implementation for bb_datasets fraud data.

Wraps the DuckDB-backed bb_datasets transaction data for use with
the air-lab-os engine. The engine sees only DatasetHandle — it never
imports from bb_datasets directly.

Split: first 80% by txn_id sort order = train, last 20% = eval.
Split is fixed and deterministic. Do not parameterise it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from functools import cached_property

import pandas as pd

# bb_datasets lives as a sibling repo — add to sys.path so it's importable
_BB_PATH = Path(__file__).parent.parent.parent.parent / "bb_datasets"
if str(_BB_PATH) not in sys.path:
    sys.path.insert(0, str(_BB_PATH))

from fraud.load import load_transactions       # noqa: E402
from fraud.features import build_features      # noqa: E402
from datasets.base import DatasetHandle, DatasetMeta


class FraudHandle(DatasetHandle):
    """
    DatasetHandle for the bb_datasets fraud dataset.

    eval_df() returns the last 20% of transactions sorted by txn_id.
    The fraud_flag column is the ground truth label.

    Feature engineering (build_features) is applied before splitting
    so both train and eval have the same engineered columns.
    """

    _LABEL_COLUMN   = "fraud_flag"
    _PRIMARY_METRIC = "f1_score"
    _SPLIT_RATIO    = 0.8          # 80% train, 20% eval — immutable

    def __init__(self, db_path: str | Path | None = None):
        """
        Args:
            db_path: path to sandbox.db. Defaults to
                     ../bb_datasets/exports/duckdb/sandbox.db
        """
        if db_path is None:
            db_path = _BB_PATH / "exports" / "duckdb" / "sandbox.db"
        self._db_path = Path(db_path)
        if not self._db_path.exists():
            raise FileNotFoundError(
                f"DuckDB not found at {self._db_path}. "
                f"Expected bb_datasets at {_BB_PATH}"
            )

    @cached_property
    def _full_df(self) -> pd.DataFrame:
        """
        Full dataset with features, sorted by txn_id for deterministic split.
        Cached — loaded once per FraudHandle instance.
        """
        raw = load_transactions(str(self._db_path))
        df  = build_features(raw)
        return df.sort_values("txn_id").reset_index(drop=True)

    @cached_property
    def _split_idx(self) -> int:
        return int(len(self._full_df) * self._SPLIT_RATIO)

    @property
    def meta(self) -> DatasetMeta:
        return DatasetMeta(
            name           = "bb_fraud_v1",
            domain         = "fraud",
            tier           = "bronze",
            version        = "1.0",
            label_column   = self._LABEL_COLUMN,
            primary_metric = self._PRIMARY_METRIC,
            row_count      = len(self._full_df),
            description    = "bb_datasets DuckDB fraud transactions, 80/20 split",
            extra          = {"db_path": str(self._db_path)},
        )

    def eval_df(self) -> pd.DataFrame:
        """Last 20% of transactions sorted by txn_id. Fixed. Immutable."""
        return self._full_df.iloc[self._split_idx:].reset_index(drop=True)

    def train_df(self) -> pd.DataFrame:
        """First 80% of transactions sorted by txn_id."""
        return self._full_df.iloc[: self._split_idx].reset_index(drop=True)

    def labels(self) -> list[bool]:
        """Ground truth for eval_df(). Same order as eval_df() rows."""
        return self.eval_df()[self._LABEL_COLUMN].astype(bool).tolist()


def get_handle(db_path: str | Path | None = None) -> FraudHandle:
    """
    Entry point for `main.py --dataset use_cases.fraud.handle`.

    Usage:
        uv run python main.py run --pattern rule_spike \
            --dataset use_cases.fraud.handle
    """
    return FraudHandle(db_path=db_path)
```

---

## Component 2: Fraud patterns

Each pattern must:
- Implement `PatternHandler` (import from `patterns.base`)
- Set `primary_metric_value` on `RunResult` to F1 score computed against
  `handle.labels()` using scikit-learn's `f1_score`
- Populate `explanation` list: one string per row in eval_df
- Expose `get_pattern()` at module level — engine discovers via this function
- Store no global state — instantiating a pattern must be side-effect-free

### F1 helper

All three patterns use the same F1 computation. Put this helper in
`use_cases/fraud/patterns/__init__.py`:

```python
"""Shared utilities for fraud patterns."""

from __future__ import annotations
from sklearn.metrics import f1_score as sklearn_f1


def compute_f1(flags: list[bool], labels: list[bool]) -> float:
    """
    Compute binary F1 score. Returns 0.0 if no positives predicted or true.
    Never raises — safe to call with all-False flags.
    """
    if not any(flags) and not any(labels):
        return 1.0   # trivial: nothing to find, found nothing
    if not any(flags) or not any(labels):
        return 0.0
    return float(sklearn_f1(labels, flags, zero_division=0))
```

---

### Pattern 1: `use_cases/fraud/patterns/rule_spike.py`

```python
"""
rule_spike — flags transactions where |amount| exceeds threshold * account_mean.

Wraps the logic from bb_datasets detect_fraud_v2 (large spike rule).
Threshold is a configurable hyperparameter — tune it across runs.

Primary metric: F1 score against fraud_flag ground truth.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_BB_PATH = Path(__file__).parent.parent.parent.parent.parent / "bb_datasets"
if str(_BB_PATH) not in sys.path:
    sys.path.insert(0, str(_BB_PATH))

from datasets.base import DatasetHandle
from patterns.base import PatternHandler, RunResult
from use_cases.fraud.patterns import compute_f1


class RuleSpike(PatternHandler):
    """
    Flag rows where |amount| > threshold × per-account mean |amount|.

    This is the core spike detection signal — isolates unusually large
    transactions relative to each account's normal spending baseline.

    Args:
        threshold: multiplier above account mean that triggers a flag.
                   Default 5.0. Tune upward to reduce false positives.
    """

    name    = "rule_spike"
    version = "0.1"

    def __init__(self, threshold: float = 5.0):
        self.threshold = threshold

    def run(self, handle: DatasetHandle) -> RunResult:
        df     = handle.eval_df()
        labels = handle.labels()
        n      = len(df)

        # Compute per-account mean absolute amount from the eval slice.
        # Note: using eval slice only — no train leakage for a rule-based pattern.
        abs_amt      = df["amount"].abs()
        account_mean = df.groupby("account_id")["amount"].transform(
            lambda x: x.abs().mean()
        )

        flags: list[bool] = []
        scores: list[float] = []
        explanation: list[str] = []

        for i in range(n):
            amt  = abs_amt.iloc[i]
            mean = account_mean.iloc[i]
            ratio = (amt / mean) if mean > 0 else 0.0
            flagged = ratio > self.threshold

            flags.append(flagged)
            scores.append(min(1.0, ratio / (self.threshold * 2)) if flagged else 0.0)
            if flagged:
                explanation.append(
                    f"amount {amt:.0f} is {ratio:.1f}x account mean {mean:.0f}"
                )
            else:
                explanation.append("")

        result = RunResult(
            flags=flags, scores=scores, explanation=explanation
        )
        result.primary_metric_value = compute_f1(flags, labels)
        result.extra_metrics = {
            "precision": _precision(flags, labels),
            "recall":    _recall(flags, labels),
            "threshold": self.threshold,
            "n_flagged": sum(flags),
        }
        return result

    def describe(self) -> dict:
        return {"pattern": self.name, "version": self.version, "threshold": self.threshold}


def _precision(flags: list[bool], labels: list[bool]) -> float:
    tp = sum(f and l for f, l in zip(flags, labels))
    fp = sum(f and not l for f, l in zip(flags, labels))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def _recall(flags: list[bool], labels: list[bool]) -> float:
    tp = sum(f and l for f, l in zip(flags, labels))
    fn = sum(not f and l for f, l in zip(flags, labels))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def get_pattern() -> RuleSpike:
    """Entry point for engine pattern discovery."""
    return RuleSpike()
```

---

### Pattern 2: `use_cases/fraud/patterns/rule_velocity.py`

```python
"""
rule_velocity — flags transactions that are part of a same-timestamp burst.

Wraps the burst detection logic from bb_datasets (velocity burst pattern):
flags DEBIT transactions when the same account has >= burst_count DEBITs
at the exact same timestamp.

Primary metric: F1 score against fraud_flag ground truth.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from datasets.base import DatasetHandle
from patterns.base import PatternHandler, RunResult
from use_cases.fraud.patterns import compute_f1


class RuleVelocity(PatternHandler):
    """
    Flag DEBIT transactions in same-timestamp bursts per account.

    A burst is defined as >= burst_count DEBIT transactions from the same
    account_id at the exact same timestamp. All transactions in the burst
    are flagged.

    Args:
        burst_count: minimum number of same-ts DEBITs to trigger.
                     Default 3. Lower = more sensitive, higher = fewer FP.
    """

    name    = "rule_velocity"
    version = "0.1"

    def __init__(self, burst_count: int = 3):
        self.burst_count = burst_count

    def run(self, handle: DatasetHandle) -> RunResult:
        df     = handle.eval_df()
        labels = handle.labels()
        n      = len(df)

        # Count DEBITs per (account_id, timestamp) group
        debit_mask = df["txn_type"] == "DEBIT"
        same_ts_counts = (
            df[debit_mask]
            .groupby(["account_id", "timestamp"])["txn_id"]
            .transform("count")
        )
        # Reindex to full df (non-DEBIT rows get NaN → fill 0)
        same_ts_counts = same_ts_counts.reindex(df.index, fill_value=0)

        flags: list[bool] = []
        scores: list[float] = []
        explanation: list[str] = []

        for i in range(n):
            count   = int(same_ts_counts.iloc[i])
            is_debit = debit_mask.iloc[i]
            flagged  = is_debit and count >= self.burst_count

            flags.append(flagged)
            scores.append(min(1.0, count / (self.burst_count * 2)) if flagged else 0.0)
            if flagged:
                explanation.append(
                    f"burst: {count} DEBITs at same timestamp for account "
                    f"{df['account_id'].iloc[i]}"
                )
            else:
                explanation.append("")

        result = RunResult(
            flags=flags, scores=scores, explanation=explanation
        )
        result.primary_metric_value = compute_f1(flags, labels)
        result.extra_metrics = {
            "precision":   _precision(flags, labels),
            "recall":      _recall(flags, labels),
            "burst_count": self.burst_count,
            "n_flagged":   sum(flags),
        }
        return result

    def describe(self) -> dict:
        return {"pattern": self.name, "version": self.version, "burst_count": self.burst_count}


def _precision(flags: list[bool], labels: list[bool]) -> float:
    tp = sum(f and l for f, l in zip(flags, labels))
    fp = sum(f and not l for f, l in zip(flags, labels))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def _recall(flags: list[bool], labels: list[bool]) -> float:
    tp = sum(f and l for f, l in zip(flags, labels))
    fn = sum(not f and l for f, l in zip(flags, labels))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def get_pattern() -> RuleVelocity:
    """Entry point for engine pattern discovery."""
    return RuleVelocity()
```

---

### Pattern 3: `use_cases/fraud/patterns/ml_logistic.py`

```python
"""
ml_logistic — logistic regression on bb_datasets engineered features.

Wraps detect_fraud_ml_honest from bb_datasets: trains on train_df(),
scores eval_df(). Uses the full feature set from build_features().

This is the only pattern that uses train_df(). It trains fresh on
every run — no model persistence in Phase 2.

Primary metric: F1 score against fraud_flag ground truth.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from datasets.base import DatasetHandle
from patterns.base import PatternHandler, RunResult
from use_cases.fraud.patterns import compute_f1


# Feature columns produced by build_features() that are safe to use
# for training. Excludes ID columns and the label itself.
FEATURE_COLS = [
    "amount",
    "abs_amount",
    "same_ts_count",
    "z_score",
    "account_zscore",
    "is_burst",
    "account_had_burst",
]


class MlLogistic(PatternHandler):
    """
    Logistic regression on bb_datasets engineered features.

    Trains on train_df(), predicts on eval_df(). StandardScaler is
    applied inside a sklearn Pipeline so no data leaks from eval to train.

    Args:
        threshold:    probability threshold for flagging. Default 0.5.
        max_iter:     logistic regression max iterations. Default 1000.
        C:            inverse regularisation strength. Default 1.0.
        class_weight: passed to LogisticRegression. Default "balanced"
                      to handle class imbalance in the fraud dataset.
    """

    name    = "ml_logistic"
    version = "0.1"

    def __init__(
        self,
        threshold:    float = 0.5,
        max_iter:     int   = 1000,
        C:            float = 1.0,
        class_weight: str   = "balanced",
    ):
        self.threshold    = threshold
        self.max_iter     = max_iter
        self.C            = C
        self.class_weight = class_weight

    def run(self, handle: DatasetHandle) -> RunResult:
        train = handle.train_df()
        eval_ = handle.eval_df()
        labels = handle.labels()
        n      = len(eval_)

        # Resolve available feature columns (guard against missing columns)
        available = [c for c in FEATURE_COLS if c in train.columns]

        X_train = train[available].fillna(0).values
        y_train = train["fraud_flag"].astype(int).values
        X_eval  = eval_[available].fillna(0).values

        # Train
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C            = self.C,
                max_iter     = self.max_iter,
                class_weight = self.class_weight,
                random_state = 42,
            )),
        ])
        pipe.fit(X_train, y_train)

        # Predict
        proba  = pipe.predict_proba(X_eval)[:, 1]   # P(fraud)
        preds  = proba >= self.threshold

        flags:       list[bool]  = preds.tolist()
        scores:      list[float] = [float(p) for p in proba]
        explanation: list[str]   = []

        for i in range(n):
            if flags[i]:
                explanation.append(
                    f"logistic P(fraud)={proba[i]:.3f} >= threshold {self.threshold}"
                )
            else:
                explanation.append("")

        result = RunResult(
            flags=flags, scores=scores, explanation=explanation
        )
        result.primary_metric_value = compute_f1(flags, labels)
        result.extra_metrics = {
            "precision":  _precision(flags, labels),
            "recall":     _recall(flags, labels),
            "threshold":  self.threshold,
            "n_flagged":  sum(flags),
            "n_features": len(available),
        }
        return result

    def describe(self) -> dict:
        return {
            "pattern":      self.name,
            "version":      self.version,
            "threshold":    self.threshold,
            "C":            self.C,
            "class_weight": self.class_weight,
        }


def _precision(flags: list[bool], labels: list[bool]) -> float:
    tp = sum(f and l for f, l in zip(flags, labels))
    fp = sum(f and not l for f, l in zip(flags, labels))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def _recall(flags: list[bool], labels: list[bool]) -> float:
    tp = sum(f and l for f, l in zip(flags, labels))
    fn = sum(not f and l for f, l in zip(flags, labels))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def get_pattern() -> MlLogistic:
    """Entry point for engine pattern discovery."""
    return MlLogistic()
```

---

## Component 3: Linking patterns into the engine

The engine discovers patterns from `patterns/scratch/`, `patterns/working/`,
`patterns/stable/`. Create symlinks so the engine can find the fraud patterns
without duplicating code.

After creating the pattern files, run from the repo root:

```bash
# Create the symlinks (run once)
cd patterns/scratch
ln -sf ../../use_cases/fraud/patterns/rule_spike.py rule_spike.py
ln -sf ../../use_cases/fraud/patterns/rule_velocity.py rule_velocity.py
ln -sf ../../use_cases/fraud/patterns/ml_logistic.py ml_logistic.py
cd ../..
```

Verify discovery works:
```bash
uv run python main.py status
# Should print: Registry is empty — no experiments run yet.
# (no crash = discovery is working)
```

---

## Component 4: Test suite

### `use_cases/fraud/tests/test_fraud_handle.py`

```python
"""
FraudHandle tests — verify the DatasetHandle contract is correctly satisfied.

These tests skip gracefully if bb_datasets is not available so the
engine test suite still passes in environments without the sibling repo.
"""

import pytest
import pandas as pd

try:
    from use_cases.fraud.handle import FraudHandle
    FRAUD_AVAILABLE = True
except (ImportError, FileNotFoundError):
    FRAUD_AVAILABLE = False

skip_if_no_fraud = pytest.mark.skipif(
    not FRAUD_AVAILABLE,
    reason="bb_datasets not available at ../bb_datasets/"
)


@skip_if_no_fraud
class TestFraudHandleMeta:
    def setup_method(self):
        self.handle = FraudHandle()

    def test_meta_domain(self):
        assert self.handle.meta.domain == "fraud"

    def test_meta_label_column(self):
        assert self.handle.meta.label_column == "fraud_flag"

    def test_meta_primary_metric(self):
        assert self.handle.meta.primary_metric == "f1_score"

    def test_meta_row_count_positive(self):
        assert self.handle.meta.row_count > 0

    def test_meta_tier(self):
        assert self.handle.meta.tier in ("bronze", "silver", "gold", "test")


@skip_if_no_fraud
class TestFraudHandleSplit:
    def setup_method(self):
        self.handle = FraudHandle()

    def test_eval_df_is_dataframe(self):
        df = self.handle.eval_df()
        assert isinstance(df, pd.DataFrame)

    def test_eval_df_has_label_column(self):
        df = self.handle.eval_df()
        assert "fraud_flag" in df.columns

    def test_train_eval_no_overlap(self):
        train = self.handle.train_df()
        eval_ = self.handle.eval_df()
        train_ids = set(train["txn_id"])
        eval_ids  = set(eval_["txn_id"])
        assert len(train_ids & eval_ids) == 0, \
            "train and eval share rows — split is not clean"

    def test_train_larger_than_eval(self):
        assert len(self.handle.train_df()) > len(self.handle.eval_df())

    def test_eval_is_approximately_20_percent(self):
        total = self.handle.meta.row_count
        eval_n = len(self.handle.eval_df())
        ratio = eval_n / total
        # Allow 1% tolerance for rounding
        assert abs(ratio - 0.20) < 0.01, f"eval ratio {ratio:.3f} != 0.20"

    def test_labels_length_matches_eval_df(self):
        labels = self.handle.labels()
        eval_  = self.handle.eval_df()
        assert len(labels) == len(eval_)

    def test_labels_are_bool(self):
        labels = self.handle.labels()
        assert all(isinstance(l, bool) for l in labels)

    def test_labels_match_fraud_flag_column(self):
        eval_   = self.handle.eval_df()
        labels  = self.handle.labels()
        col_bools = eval_["fraud_flag"].astype(bool).tolist()
        assert labels == col_bools

    def test_eval_df_is_deterministic(self):
        """Same rows every call — split is fixed."""
        df1 = self.handle.eval_df()
        df2 = self.handle.eval_df()
        assert list(df1["txn_id"]) == list(df2["txn_id"])

    def test_some_fraud_in_eval(self):
        """Dataset should have at least some fraud in the eval split."""
        labels = self.handle.labels()
        assert any(labels), "No fraud in eval split — check dataset"
```

### `use_cases/fraud/tests/test_fraud_patterns.py`

```python
"""
Fraud pattern tests — verify PatternHandler contract for all three patterns.

Tests use a minimal FraudHandle so they pass without bb_datasets when
possible. The integration tests (bb_datasets required) are skipped if the
database is unavailable.
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock

from datasets.base import DatasetHandle, DatasetMeta
from patterns.base import RunResult

try:
    from use_cases.fraud.handle import FraudHandle
    FRAUD_AVAILABLE = True
except (ImportError, FileNotFoundError):
    FRAUD_AVAILABLE = False

skip_if_no_fraud = pytest.mark.skipif(
    not FRAUD_AVAILABLE,
    reason="bb_datasets not available"
)


# ---------------------------------------------------------------------------
# Shared stub for unit tests (no db needed)
# ---------------------------------------------------------------------------

def _make_stub_df(n=20, fraud_rate=0.2):
    """Minimal DataFrame that satisfies FraudHandle column expectations."""
    import numpy as np
    rng = np.random.default_rng(42)
    n_fraud = max(1, int(n * fraud_rate))
    fraud_idx = rng.choice(n, size=n_fraud, replace=False)
    fraud_flags = [i in fraud_idx for i in range(n)]

    amounts = rng.uniform(100, 5000, size=n)
    # Make a couple of spike amounts
    amounts[fraud_idx[:2]] = amounts.mean() * 10

    timestamps = [f"2024-01-01 10:0{i%10}:00" for i in range(n)]
    account_ids = [f"A{(i % 5):03d}" for i in range(n)]
    txn_types = ["DEBIT" if i % 3 != 0 else "CREDIT" for i in range(n)]

    df = pd.DataFrame({
        "txn_id":            [f"T{i:05d}" for i in range(n)],
        "account_id":        account_ids,
        "amount":            amounts,
        "abs_amount":        amounts,
        "txn_type":          txn_types,
        "merchant_category": ["Sales"] * n,
        "timestamp":         timestamps,
        "fraud_flag":        fraud_flags,
        "same_ts_count":     [1] * n,
        "z_score":           [0.0] * n,
        "account_zscore":    [0.0] * n,
        "is_burst":          [False] * n,
        "account_had_burst": [False] * n,
    })
    return df


class StubFraudHandle(DatasetHandle):
    """Minimal handle for pattern unit tests — no db required."""

    def __init__(self, n=20):
        self._df = _make_stub_df(n)

    @property
    def meta(self):
        return DatasetMeta(
            name="stub_fraud", domain="fraud", tier="test", version="0.1",
            label_column="fraud_flag", primary_metric="f1_score",
            row_count=len(self._df),
        )

    def eval_df(self):
        return self._df.copy()

    def train_df(self):
        return self._df.copy()  # same data for stub — ok for unit tests

    def labels(self):
        return self._df["fraud_flag"].astype(bool).tolist()


# ---------------------------------------------------------------------------
# Contract tests — run for all three patterns via parametrize
# ---------------------------------------------------------------------------

def _get_all_patterns():
    from use_cases.fraud.patterns.rule_spike    import get_pattern as spike
    from use_cases.fraud.patterns.rule_velocity import get_pattern as velocity
    from use_cases.fraud.patterns.ml_logistic   import get_pattern as logistic
    return [spike(), velocity(), logistic()]


@pytest.mark.parametrize("pattern", _get_all_patterns())
class TestPatternContract:
    """
    Every fraud pattern must satisfy the PatternHandler contract.
    These tests use StubFraudHandle — no bb_datasets required.
    """

    def test_has_name(self, pattern):
        assert isinstance(pattern.name, str) and pattern.name

    def test_has_version(self, pattern):
        assert isinstance(pattern.version, str)

    def test_has_describe(self, pattern):
        d = pattern.describe()
        assert isinstance(d, dict)
        assert "pattern" in d

    def test_detect_returns_run_result(self, pattern):
        handle = StubFraudHandle()
        result = pattern.detect(handle)
        assert isinstance(result, RunResult)

    def test_flags_length_matches_eval_df(self, pattern):
        handle = StubFraudHandle()
        result = pattern.detect(handle)
        assert len(result.flags) == len(handle.eval_df())

    def test_scores_same_length_as_flags(self, pattern):
        handle = StubFraudHandle()
        result = pattern.detect(handle)
        assert len(result.scores) == len(result.flags)

    def test_explanation_same_length_as_flags(self, pattern):
        handle = StubFraudHandle()
        result = pattern.detect(handle)
        assert len(result.explanation) == len(result.flags)

    def test_scores_in_range(self, pattern):
        handle = StubFraudHandle()
        result = pattern.detect(handle)
        assert all(0.0 <= s <= 1.0 for s in result.scores)

    def test_primary_metric_in_range(self, pattern):
        handle = StubFraudHandle()
        result = pattern.detect(handle)
        assert 0.0 <= result.primary_metric_value <= 1.0

    def test_flagged_rows_have_explanation(self, pattern):
        handle = StubFraudHandle()
        result = pattern.detect(handle)
        for i, flagged in enumerate(result.flags):
            if flagged:
                assert result.explanation[i].strip(), \
                    f"Flagged row {i} has empty explanation"

    def test_latency_ms_set_by_base(self, pattern):
        """latency_ms should be set by PatternHandler.detect(), not run()."""
        handle = StubFraudHandle()
        result = pattern.detect(handle)
        assert result.latency_ms > 0.0

    def test_detect_is_deterministic(self, pattern):
        """Same input → same flags output."""
        handle = StubFraudHandle()
        r1 = pattern.detect(handle)
        r2 = pattern.detect(handle)
        assert r1.flags == r2.flags


# ---------------------------------------------------------------------------
# Integration tests — require bb_datasets
# ---------------------------------------------------------------------------

@skip_if_no_fraud
class TestPatternIntegration:
    """Run each pattern against the real FraudHandle."""

    def _run(self, pattern):
        handle = FraudHandle()
        return pattern.detect(handle)

    def test_rule_spike_real_data(self):
        from use_cases.fraud.patterns.rule_spike import get_pattern
        result = self._run(get_pattern())
        assert result.primary_metric_value >= 0.0
        assert any(result.flags), "Spike pattern flagged nothing on real data"

    def test_rule_velocity_real_data(self):
        from use_cases.fraud.patterns.rule_velocity import get_pattern
        result = self._run(get_pattern())
        assert result.primary_metric_value >= 0.0

    def test_ml_logistic_real_data(self):
        from use_cases.fraud.patterns.ml_logistic import get_pattern
        result = self._run(get_pattern())
        assert result.primary_metric_value >= 0.0
        assert any(result.flags), "Logistic flagged nothing on real data"

    def test_ml_logistic_trains_on_train_not_eval(self):
        """Smoke test: training on train_df() doesn't crash."""
        from use_cases.fraud.patterns.ml_logistic import get_pattern
        handle = FraudHandle()
        # This exercises both train_df() and eval_df() paths
        result = get_pattern().detect(handle)
        assert len(result.flags) == len(handle.labels())


# ---------------------------------------------------------------------------
# F1 helper tests — no db required
# ---------------------------------------------------------------------------

class TestComputeF1:
    def test_perfect_precision_recall(self):
        from use_cases.fraud.patterns import compute_f1
        flags  = [True, False, True, False]
        labels = [True, False, True, False]
        assert compute_f1(flags, labels) == pytest.approx(1.0)

    def test_all_false_no_fraud(self):
        from use_cases.fraud.patterns import compute_f1
        flags  = [False, False, False]
        labels = [False, False, False]
        assert compute_f1(flags, labels) == pytest.approx(1.0)

    def test_all_missed(self):
        from use_cases.fraud.patterns import compute_f1
        flags  = [False, False, False]
        labels = [True,  True,  False]
        assert compute_f1(flags, labels) == pytest.approx(0.0)

    def test_partial_f1(self):
        from use_cases.fraud.patterns import compute_f1
        flags  = [True, True,  False, False]
        labels = [True, False, True,  False]
        # precision=0.5, recall=0.5 → F1=0.5
        assert compute_f1(flags, labels) == pytest.approx(0.5)
```

---

## Component 5: Smoke test the full pipeline

After all files are written and symlinks created, run this to verify
the fraud plugin works end-to-end with the engine:

```bash
# Unit + integration tests (skips gracefully if bb_datasets absent)
uv run pytest use_cases/fraud/tests/ -q

# Engine status (verifies pattern discovery works)
uv run python main.py status

# Run one pattern through the engine
uv run python main.py run \
    --pattern rule_spike \
    --dataset use_cases.fraud.handle \
    --description "Phase 2 spike baseline"

# Run all three patterns and compare
uv run python main.py arena --dataset use_cases.fraud.handle

# Check registry updated
uv run python main.py status
```

Expected output from arena:
```
--- Arena results (bb_fraud_v1) ---
  1. ml_logistic                   score=0.XXXX
  2. rule_spike                    score=0.XXXX
  3. rule_velocity                 score=0.XXXX

  Winner: ml_logistic
```

The exact scores depend on the dataset. `ml_logistic` typically wins
but this is not guaranteed — it depends on hyperparameters and data.

---

## Deliverables checklist

- [ ] `use_cases/__init__.py` exists
- [ ] `use_cases/fraud/__init__.py` exists
- [ ] `use_cases/fraud/handle.py` — `FraudHandle` + `get_handle()`
- [ ] `use_cases/fraud/patterns/__init__.py` — `compute_f1()`
- [ ] `use_cases/fraud/patterns/rule_spike.py` — `RuleSpike` + `get_pattern()`
- [ ] `use_cases/fraud/patterns/rule_velocity.py` — `RuleVelocity` + `get_pattern()`
- [ ] `use_cases/fraud/patterns/ml_logistic.py` — `MlLogistic` + `get_pattern()`
- [ ] `use_cases/fraud/tests/__init__.py` exists
- [ ] `use_cases/fraud/tests/test_fraud_handle.py`
- [ ] `use_cases/fraud/tests/test_fraud_patterns.py`
- [ ] Symlinks: `patterns/scratch/rule_spike.py`, `rule_velocity.py`, `ml_logistic.py`
- [ ] `uv run pytest tests/ -q` still passes (engine tests unchanged)
- [ ] `uv run pytest use_cases/fraud/tests/ -q` passes
- [ ] `uv run python main.py arena --dataset use_cases.fraud.handle` runs without error
- [ ] `uv run python main.py status` shows three patterns with scores

## What Claude Code must NOT do in Phase 2

- Do not modify any file in `runtime/`, `lab/`, `evaluation/`, `memory/`,
  `datasets/base.py`, `patterns/base.py`, `main.py`, or `scoring_policy.yaml`
- Do not copy `bb_datasets` code into this repo — import it via sys.path
- Do not add new top-level dependencies to `pyproject.toml` (scikit-learn
  and pandas are already installed)
- Do not create a `registry.json` or `memory/runs.json` manually — the
  engine creates these on first run
- Do not hardcode paths to `sandbox.db` using absolute paths — use the
  relative `_BB_PATH` pattern shown in `handle.py`

---

*End of Phase 2 spec.*
*Phase 3: dashboard + SSE log stream.*
*Phase 4: auto-promote loop + notifications.*
