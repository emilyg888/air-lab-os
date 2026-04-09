# SPEC_PRIMARY_METRIC.md — Fix Primary Metric Ownership

> **For Claude Code.** Read CLAUDE.md, then SPEC.md, then this file.
> Phase 1 must be passing (`uv run pytest tests/ -q`) before starting here.
>
> This spec fixes a fundamental contract violation in the evaluation
> pipeline. It touches four files. Read the entire spec before
> writing a single line of code.

---

## The problem in one paragraph

The evaluation system has a broken chain of authority. `DatasetMeta`
declares `primary_metric: "f1_score"` — which metric defines "good" for
this dataset. But the evaluator never reads it. Instead, patterns compute
their own score and write it to `RunResult.primary_metric_value`, and the
evaluator blindly trusts whatever number the pattern put there. The thing
being judged is judging itself. The dataset's declaration is decorative.
This spec fixes the authority chain so it runs in one direction:

```
Dataset declares → Evaluator computes → Pattern cannot influence
```

---

## Exact changes — four files, nothing else

### 1. `patterns/base.py` — remove `primary_metric_value` from `RunResult`

`RunResult` is the pattern's output contract. Its job is to carry
predictions: `flags`, `scores`, `explanation`. It is not responsible
for computing or carrying a performance metric — that is the evaluator's
job. Remove `primary_metric_value` entirely.

`extra_metrics` stays. Patterns may still report supplementary
diagnostic numbers (precision, recall, threshold, n_flagged) for
logging. These go into `runs.json` under `extra` and are never used
to compute the composite score.

**Rewrite `patterns/base.py` to exactly this:**

```python
"""
PatternHandler — abstract contract for all pattern plugins.

Every pattern the engine runs must implement this interface.
The engine calls detect() and never imports domain-specific code.

Pattern files live in patterns/scratch/, patterns/working/, or
patterns/stable/ depending on their current promotion tier.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from datasets.base import DatasetHandle


@dataclass
class RunResult:
    """
    Standardised output contract for every pattern run.

    All three lists must be the same length as eval_df().

    flags       — True if the row is a positive detection.
    scores      — float in [0.0, 1.0], per-row confidence or probability.
                  Higher = more likely to be a true positive.
    explanation — human-readable reason string per row. Empty string ""
                  if the row is not flagged or no explanation is available.
    latency_ms  — wall-clock milliseconds for the full detect() call.
                  Set by PatternHandler.detect(). Do NOT set in run().
    cost_per_1k — estimated USD cost per 1000 rows. 0.0 for local models.
                  Set this if the pattern calls an external API.
    extra_metrics — optional dict of supplementary diagnostic values for
                  logging (e.g. {"precision": 0.8, "n_flagged": 12}).
                  These are written to runs.json but never affect scoring.

    What patterns must NOT do:
      - Do not compute or set any performance metric (F1, accuracy, etc).
        The evaluator computes all metrics from flags and ground truth.
        The dataset declares which metric to use via DatasetMeta.primary_metric.
      - Do not set latency_ms. The base class sets it automatically.
    """
    flags:        list[bool]
    scores:       list[float]
    explanation:  list[str]
    latency_ms:   float = 0.0
    cost_per_1k:  float = 0.0
    extra_metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.flags)
        assert len(self.scores) == n, \
            f"scores length {len(self.scores)} must match flags length {n}"
        assert len(self.explanation) == n, \
            f"explanation length {len(self.explanation)} must match flags length {n}"
        assert all(0.0 <= s <= 1.0 for s in self.scores), \
            "all scores must be in [0.0, 1.0]"


class PatternHandler(ABC):
    """
    Abstract base for all pattern implementations.

    Subclasses implement run(). The base class wraps run() to record
    wall-clock latency automatically via detect().

    Pattern contract — a pattern is responsible for:
      1. Reading handle.eval_df() to get the rows to score.
      2. Producing flags, scores, explanation of the same length.
      3. Optionally reading handle.train_df() for ML patterns that fit.
      4. Optionally populating extra_metrics for diagnostic logging.

    A pattern is NOT responsible for:
      - Computing F1, accuracy, or any performance metric.
      - Knowing which metric the dataset uses.
      - Setting primary_metric_value (field removed — evaluator owns this).

    Usage:
        handle  = MyDatasetHandle()
        pattern = MyPattern(threshold=0.5)
        result  = pattern.detect(handle)   # latency_ms set automatically
    """

    name:    str = "base"    # unique pattern identifier — override in subclass
    version: str = "0.1"    # pattern version string — override in subclass

    def detect(self, handle: DatasetHandle) -> RunResult:
        """
        Public entry point. Times the call and injects latency_ms.
        Do NOT override — override run() instead.
        """
        t0 = time.perf_counter()
        result = self.run(handle)
        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    @abstractmethod
    def run(self, handle: DatasetHandle) -> RunResult:
        """
        Implement detection logic here.

        Args:
            handle: DatasetHandle providing eval_df(), train_df(), labels().
                    Do not assume specific column names beyond what
                    DatasetMeta declares.

        Returns:
            RunResult with flags, scores, explanation — same length as
            handle.eval_df(). Do NOT set latency_ms. Do NOT compute
            or set any performance metric.
        """
        ...

    def describe(self) -> dict:
        """
        Return hyperparameter dict for run logging.
        Override in subclasses to include pattern-specific config.
        """
        return {"pattern": self.name, "version": self.version}
```

---

### 2. `evaluation/evaluator.py` — compute primary metric from flags + labels + dataset declaration

The evaluator now owns the full scoring pipeline. It reads
`handle.meta.primary_metric` to know which metric to compute, then
computes it deterministically from `result.flags` and `handle.labels()`.
The pattern has no input into this computation.

Supported metric names — the evaluator dispatches on these strings:

| `primary_metric` value | Computation |
|------------------------|-------------|
| `"f1_score"`           | sklearn `f1_score(labels, flags, zero_division=0)` |
| `"precision"`          | sklearn `precision_score(labels, flags, zero_division=0)` |
| `"recall"`             | sklearn `recall_score(labels, flags, zero_division=0)` |
| `"accuracy"`           | sklearn `accuracy_score(labels, flags)` |
| `"average_precision"`  | sklearn `average_precision_score(labels, scores)` — uses `result.scores`, not flags |

If the dataset declares an unknown metric name, the evaluator raises
`ValueError` immediately with the full list of valid names. This fails
loud and early — not silently at score time.

**Rewrite `evaluation/evaluator.py` to exactly this:**

```python
"""
Evaluator — locked scoring layer.

evaluate(result, handle, policy) → EvalMetrics

Authority chain:
  Dataset  declares → which metric defines "good" (primary_metric name)
  Evaluator computes → the metric value from flags and ground truth
  Pattern  cannot    → set or influence the primary metric

The pattern's job ends at RunResult.flags/scores/explanation.
Everything else is the evaluator's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from datasets.base import DatasetHandle
from patterns.base import RunResult


POLICY_PATH = Path(__file__).parent.parent / "scoring_policy.yaml"


# ---------------------------------------------------------------------------
# Supported primary metrics — dataset declares one of these names.
# Add new metrics here. The key is what goes in DatasetMeta.primary_metric.
# ---------------------------------------------------------------------------

def _f1(flags, labels, scores):
    if not any(flags):
        return 0.0
    return float(f1_score(labels, flags, zero_division=0))


def _precision(flags, labels, scores):
    if not any(flags):
        return 0.0
    return float(precision_score(labels, flags, zero_division=0))


def _recall(flags, labels, scores):
    return float(recall_score(labels, flags, zero_division=0))


def _accuracy(flags, labels, scores):
    return float(accuracy_score(labels, flags))


def _average_precision(flags, labels, scores):
    """Uses continuous scores, not binary flags — rewards calibrated confidence."""
    if not any(labels):
        return 0.0
    return float(average_precision_score(labels, scores))


SUPPORTED_METRICS: dict[str, Any] = {
    "f1_score":          _f1,
    "precision":         _precision,
    "recall":            _recall,
    "accuracy":          _accuracy,
    "average_precision": _average_precision,
}


def compute_primary_metric(
    metric_name: str,
    flags:       list[bool],
    labels:      list[bool],
    scores:      list[float],
) -> float:
    """
    Compute the primary metric for a run.

    Args:
        metric_name: value of DatasetMeta.primary_metric — e.g. "f1_score"
        flags:       binary predictions from RunResult.flags
        labels:      ground truth from DatasetHandle.labels()
        scores:      continuous predictions from RunResult.scores

    Returns:
        float in [0.0, 1.0]

    Raises:
        ValueError: if metric_name is not in SUPPORTED_METRICS
    """
    if metric_name not in SUPPORTED_METRICS:
        raise ValueError(
            f"Unknown primary_metric '{metric_name}'. "
            f"Valid values: {sorted(SUPPORTED_METRICS.keys())}. "
            f"Check DatasetMeta.primary_metric in your DatasetHandle "
            f"or datasets/metadata.json."
        )
    fn = SUPPORTED_METRICS[metric_name]
    y_true  = [int(l) for l in labels]
    y_pred  = [int(f) for f in flags]
    y_score = [float(s) for s in scores]
    value   = fn(y_pred, y_true, y_score)
    return max(0.0, min(1.0, float(value)))


# ---------------------------------------------------------------------------
# Scoring policy
# ---------------------------------------------------------------------------

@dataclass
class ScoringPolicy:
    primary_metric_weight: float
    explainability_weight: float
    latency_weight:        float
    cost_weight:           float
    latency_max_ms:        float
    cost_max_per_1k:       float
    working_threshold:     float
    stable_threshold:      float
    min_runs:              int

    def validate(self) -> None:
        total = (
            self.primary_metric_weight
            + self.explainability_weight
            + self.latency_weight
            + self.cost_weight
        )
        assert abs(total - 1.0) < 1e-6, (
            f"Weights must sum to 1.0, got {total:.6f}. "
            f"Check scoring_policy.yaml."
        )


def load_policy(path: Path = POLICY_PATH) -> ScoringPolicy:
    """Load and validate the scoring policy. Raises on bad weights."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    w = raw["weights"]
    policy = ScoringPolicy(
        primary_metric_weight = w["primary_metric"],
        explainability_weight = w["explainability"],
        latency_weight        = w["latency"],
        cost_weight           = w["cost"],
        latency_max_ms        = raw["latency"]["max_ms"],
        cost_max_per_1k       = raw["cost"]["max_per_1k"],
        working_threshold     = raw["promotion"]["working_threshold"],
        stable_threshold      = raw["promotion"]["stable_threshold"],
        min_runs              = raw["promotion"]["min_runs"],
    )
    policy.validate()
    return policy


# ---------------------------------------------------------------------------
# Evaluation output
# ---------------------------------------------------------------------------

@dataclass
class EvalMetrics:
    """
    Full scoring breakdown for one experiment run.

    primary_metric_name  — which metric was used (from DatasetMeta)
    primary_metric_value — computed by evaluator from flags + ground truth
    explainability_score — fraction of flagged rows with non-empty explanation
    latency_score        — 1.0 = instant, 0.0 = latency_max_ms or slower
    cost_score           — 1.0 = free, 0.0 = cost_max_per_1k or more expensive
    score                — weighted composite [0.0, 1.0] — drives promotion
    extra                — supplementary metrics from RunResult.extra_metrics
    """
    primary_metric_name:  str
    primary_metric_value: float
    explainability_score: float
    latency_score:        float
    cost_score:           float
    score:                float
    extra:                dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    result: RunResult,
    handle: DatasetHandle,
    policy: ScoringPolicy | None = None,
) -> EvalMetrics:
    """
    Score a RunResult against ground truth.

    The primary metric is determined by handle.meta.primary_metric and
    computed from result.flags and handle.labels(). The pattern has no
    input into this computation.

    Args:
        result: output of PatternHandler.detect()
        handle: DatasetHandle — provides labels() and meta.primary_metric
        policy: loaded ScoringPolicy; loads from yaml if None

    Returns:
        EvalMetrics with all dimension scores and the weighted composite

    Raises:
        AssertionError: if result.flags length != handle.labels() length
        ValueError: if handle.meta.primary_metric is not a supported metric
    """
    if policy is None:
        policy = load_policy()

    labels = handle.labels()
    assert len(result.flags) == len(labels), (
        f"RunResult has {len(result.flags)} flags "
        f"but dataset has {len(labels)} labels"
    )

    # --- Primary metric — evaluator computes, dataset declares, pattern cannot set ---
    metric_name = handle.meta.primary_metric
    primary_value = compute_primary_metric(
        metric_name = metric_name,
        flags       = result.flags,
        labels      = labels,
        scores      = result.scores,
    )

    # --- Explainability ---
    # Fraction of flagged rows that have a non-empty explanation string.
    # If nothing is flagged: 1.0 (nothing to explain — not penalised).
    flagged = [i for i, f in enumerate(result.flags) if f]
    if flagged:
        explained    = sum(1 for i in flagged if result.explanation[i].strip())
        expl_score   = explained / len(flagged)
    else:
        expl_score   = 1.0

    # --- Latency ---
    latency_score = max(0.0, 1.0 - result.latency_ms / policy.latency_max_ms)

    # --- Cost ---
    cost_score = max(0.0, 1.0 - result.cost_per_1k / policy.cost_max_per_1k)

    # --- Weighted composite ---
    composite = (
        policy.primary_metric_weight * primary_value
        + policy.explainability_weight * expl_score
        + policy.latency_weight        * latency_score
        + policy.cost_weight           * cost_score
    )

    return EvalMetrics(
        primary_metric_name  = metric_name,
        primary_metric_value = round(primary_value, 4),
        explainability_score = round(expl_score, 4),
        latency_score        = round(latency_score, 4),
        cost_score           = round(cost_score, 4),
        score                = round(composite, 4),
        extra                = result.extra_metrics or {},
    )
```

---

### 3. `tests/test_evaluator.py` — rewrite for the new contract

The existing tests are wrong in two ways. First, they set
`r.primary_metric_value = primary` on `RunResult` — that field no longer
exists. Second, they test that the evaluator passes through a caller-supplied
value, which is exactly the self-scoring bug being fixed. The new tests
verify that the evaluator computes the metric correctly from flags and labels,
and that it respects the dataset's declared metric name.

**Rewrite `tests/test_evaluator.py` to exactly this:**

```python
"""
Evaluator tests — verifies the authority chain is correct.

Key invariants under test:
  1. The evaluator reads handle.meta.primary_metric — not the pattern's output.
  2. The evaluator computes the metric from flags + ground truth.
  3. Different metric names produce different (correct) values.
  4. Unknown metric names raise ValueError.
  5. Explainability, latency, cost dimensions work as before.
  6. Weights sum to 1.0.
"""

import pytest
import pandas as pd
from evaluation.evaluator import (
    evaluate,
    load_policy,
    compute_primary_metric,
    EvalMetrics,
    SUPPORTED_METRICS,
)
from patterns.base import RunResult
from datasets.base import DatasetHandle, DatasetMeta


# ---------------------------------------------------------------------------
# Shared stubs
# ---------------------------------------------------------------------------

class StubHandle(DatasetHandle):
    """Handle with 3 positives and 7 negatives. primary_metric is configurable."""

    def __init__(self, primary_metric: str = "f1_score", n=10, n_pos=3):
        self._metric = primary_metric
        self._n      = n
        self._n_pos  = n_pos

    @property
    def meta(self) -> DatasetMeta:
        return DatasetMeta(
            name="stub", domain="test", tier="test", version="0.1",
            label_column="label",
            primary_metric=self._metric,
            row_count=self._n,
        )

    def eval_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "label": [True] * self._n_pos + [False] * (self._n - self._n_pos)
        })

    def train_df(self) -> pd.DataFrame:
        return self.eval_df()

    def labels(self) -> list[bool]:
        return [True] * self._n_pos + [False] * (self._n - self._n_pos)


def _perfect_result(n=10, n_pos=3) -> RunResult:
    """Flags exactly the positives. Perfect precision and recall."""
    return RunResult(
        flags       = [True]  * n_pos + [False] * (n - n_pos),
        scores      = [0.95]  * n_pos + [0.05]  * (n - n_pos),
        explanation = ["flagged"] * n_pos + [""] * (n - n_pos),
    )


def _empty_result(n=10) -> RunResult:
    """Flags nothing."""
    return RunResult(
        flags       = [False] * n,
        scores      = [0.0]   * n,
        explanation = [""]    * n,
    )


def _all_flagged_result(n=10) -> RunResult:
    """Flags everything."""
    return RunResult(
        flags       = [True] * n,
        scores      = [0.9]  * n,
        explanation = ["flagged"] * n,
    )


@pytest.fixture
def policy():
    return load_policy()


# ---------------------------------------------------------------------------
# Authority chain: evaluator computes metric from flags + labels, not from pattern
# ---------------------------------------------------------------------------

class TestAuthorityChain:
    """
    The core invariant: the evaluator owns metric computation.
    The pattern's RunResult carries no metric value — only flags and scores.
    """

    def test_runresult_has_no_primary_metric_value_field(self):
        """RunResult must not have primary_metric_value — field was removed."""
        r = RunResult(flags=[True], scores=[0.9], explanation=["x"])
        assert not hasattr(r, "primary_metric_value"), (
            "primary_metric_value field still exists on RunResult. "
            "It must be removed — the evaluator computes this, not the pattern."
        )

    def test_evaluator_reads_handle_metric_name(self, policy):
        """EvalMetrics reports which metric name was used, sourced from the handle."""
        handle = StubHandle(primary_metric="f1_score")
        result = _perfect_result()
        m = evaluate(result, handle, policy)
        assert m.primary_metric_name == "f1_score"

    def test_evaluator_uses_dataset_metric_not_pattern_output(self, policy):
        """
        Two handles with different primary_metric declarations.
        Same flags → different primary_metric_value because the dataset
        controls which function is applied, not the pattern.
        """
        result = _perfect_result()   # flags = [T,T,T,F,F,F,F,F,F,F]
        labels = [True]*3 + [False]*7

        f1_handle  = StubHandle(primary_metric="f1_score")
        acc_handle = StubHandle(primary_metric="accuracy")

        m_f1  = evaluate(result, f1_handle,  policy)
        m_acc = evaluate(result, acc_handle, policy)

        # Both should be high for a perfect result, but via different computations
        assert m_f1.primary_metric_name  == "f1_score"
        assert m_acc.primary_metric_name == "accuracy"
        # F1 on 3/10 positives = 1.0 (perfect)
        assert m_f1.primary_metric_value  == pytest.approx(1.0)
        # Accuracy on 3/10 positives = 1.0 (all correct)
        assert m_acc.primary_metric_value == pytest.approx(1.0)

    def test_perfect_flags_give_perfect_f1(self, policy):
        handle = StubHandle(primary_metric="f1_score")
        m = evaluate(_perfect_result(), handle, policy)
        assert m.primary_metric_value == pytest.approx(1.0)

    def test_empty_flags_give_zero_f1(self, policy):
        handle = StubHandle(primary_metric="f1_score")
        m = evaluate(_empty_result(), handle, policy)
        assert m.primary_metric_value == pytest.approx(0.0)

    def test_empty_flags_give_zero_recall(self, policy):
        handle = StubHandle(primary_metric="recall")
        m = evaluate(_empty_result(), handle, policy)
        assert m.primary_metric_value == pytest.approx(0.0)

    def test_all_flagged_with_some_positives_gives_partial_f1(self, policy):
        """Flags all 10 rows but only 3 are truly positive → low precision."""
        handle = StubHandle(primary_metric="f1_score", n=10, n_pos=3)
        result = _all_flagged_result(n=10)
        m = evaluate(result, handle, policy)
        # precision=3/10=0.3, recall=1.0 → F1 = 2*0.3/(1.3) ≈ 0.46
        assert 0.40 < m.primary_metric_value < 0.55

    def test_unknown_metric_raises_value_error(self, policy):
        handle = StubHandle(primary_metric="roc_auc")
        with pytest.raises(ValueError, match="Unknown primary_metric"):
            evaluate(_perfect_result(), handle, policy)

    def test_unknown_metric_error_lists_valid_names(self, policy):
        handle = StubHandle(primary_metric="invented_metric")
        with pytest.raises(ValueError) as exc_info:
            evaluate(_perfect_result(), handle, policy)
        msg = str(exc_info.value)
        for name in SUPPORTED_METRICS:
            assert name in msg


# ---------------------------------------------------------------------------
# compute_primary_metric unit tests
# ---------------------------------------------------------------------------

class TestComputePrimaryMetric:
    """Unit tests for the metric dispatcher — isolated from evaluate()."""

    def test_f1_perfect(self):
        flags  = [True,  True,  False, False]
        labels = [True,  True,  False, False]
        scores = [0.9,   0.9,   0.1,   0.1]
        assert compute_primary_metric("f1_score", flags, labels, scores) == pytest.approx(1.0)

    def test_f1_zero_when_nothing_flagged(self):
        flags  = [False, False, False]
        labels = [True,  False, False]
        scores = [0.0,   0.0,   0.0]
        assert compute_primary_metric("f1_score", flags, labels, scores) == pytest.approx(0.0)

    def test_precision_perfect(self):
        flags  = [True,  False, False]
        labels = [True,  True,  False]
        scores = [0.9,   0.1,   0.1]
        # precision = 1/1 = 1.0 (everything flagged is correct)
        assert compute_primary_metric("precision", flags, labels, scores) == pytest.approx(1.0)

    def test_recall_perfect(self):
        flags  = [True,  True,  False]
        labels = [True,  True,  False]
        scores = [0.9,   0.9,   0.1]
        assert compute_primary_metric("recall", flags, labels, scores) == pytest.approx(1.0)

    def test_accuracy_perfect(self):
        flags  = [True, False, False]
        labels = [True, False, False]
        scores = [0.9,  0.1,   0.1]
        assert compute_primary_metric("accuracy", flags, labels, scores) == pytest.approx(1.0)

    def test_accuracy_partial(self):
        flags  = [True, True,  False]
        labels = [True, False, False]
        scores = [0.9,  0.9,   0.1]
        # 2/3 correct
        assert compute_primary_metric("accuracy", flags, labels, scores) == pytest.approx(2/3)

    def test_average_precision_uses_scores_not_flags(self):
        """average_precision should differ from binary F1 when scores vary."""
        flags  = [True,  True,  False, False]
        labels = [True,  False, True,  False]
        scores = [0.95,  0.4,   0.6,   0.1]
        ap = compute_primary_metric("average_precision", flags, labels, scores)
        # AP is based on scores, not flags — just verify it's a valid float
        assert 0.0 <= ap <= 1.0

    def test_all_supported_metrics_return_float_in_range(self):
        flags  = [True, False, True,  False]
        labels = [True, True,  False, False]
        scores = [0.9,  0.4,   0.6,   0.1]
        for name in SUPPORTED_METRICS:
            val = compute_primary_metric(name, flags, labels, scores)
            assert 0.0 <= val <= 1.0, \
                f"Metric '{name}' returned {val} — out of [0, 1]"

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown primary_metric"):
            compute_primary_metric("invented", [True], [True], [0.9])


# ---------------------------------------------------------------------------
# Explainability, latency, cost — unchanged behaviour
# ---------------------------------------------------------------------------

class TestOtherDimensions:
    def test_full_explainability_when_all_flagged_rows_have_text(self, policy):
        handle = StubHandle(primary_metric="f1_score")
        result = _perfect_result()
        m = evaluate(result, handle, policy)
        assert m.explainability_score == pytest.approx(1.0)

    def test_zero_explainability_when_no_explanation(self, policy):
        handle = StubHandle(primary_metric="f1_score")
        result = RunResult(
            flags       = [True]*3 + [False]*7,
            scores      = [0.9]*3  + [0.1]*7,
            explanation = [""]    * 10,   # flagged rows have no explanation
        )
        m = evaluate(result, handle, policy)
        assert m.explainability_score == pytest.approx(0.0)

    def test_nothing_flagged_gives_explainability_one(self, policy):
        """Nothing to explain → not penalised."""
        handle = StubHandle(primary_metric="f1_score")
        m = evaluate(_empty_result(), handle, policy)
        assert m.explainability_score == pytest.approx(1.0)

    def test_latency_penalised(self, policy):
        handle = StubHandle(primary_metric="f1_score")
        fast = RunResult(flags=[False]*10, scores=[0.0]*10, explanation=[""]*10,
                         latency_ms=0.0)
        slow = RunResult(flags=[False]*10, scores=[0.0]*10, explanation=[""]*10,
                         latency_ms=500.0)
        m_fast = evaluate(fast, handle, policy)
        m_slow = evaluate(slow, handle, policy)
        assert m_fast.latency_score > m_slow.latency_score
        assert m_slow.latency_score == pytest.approx(0.0)

    def test_cost_penalised(self, policy):
        handle = StubHandle(primary_metric="f1_score")
        free      = RunResult(flags=[False]*10, scores=[0.0]*10, explanation=[""]*10,
                              cost_per_1k=0.0)
        expensive = RunResult(flags=[False]*10, scores=[0.0]*10, explanation=[""]*10,
                              cost_per_1k=0.10)
        m_free = evaluate(free,      handle, policy)
        m_exp  = evaluate(expensive, handle, policy)
        assert m_free.cost_score > m_exp.cost_score
        assert m_exp.cost_score == pytest.approx(0.0)

    def test_score_in_range(self, policy):
        handle = StubHandle(primary_metric="f1_score")
        m = evaluate(_perfect_result(), handle, policy)
        assert 0.0 <= m.score <= 1.0

    def test_weights_sum_to_one(self, policy):
        total = (
            policy.primary_metric_weight
            + policy.explainability_weight
            + policy.latency_weight
            + policy.cost_weight
        )
        assert abs(total - 1.0) < 1e-6

    def test_extra_metrics_passed_through(self, policy):
        handle = StubHandle(primary_metric="f1_score")
        result = _perfect_result()
        result.extra_metrics = {"n_flagged": 3, "threshold": 0.5}
        m = evaluate(result, handle, policy)
        assert m.extra["n_flagged"] == 3
        assert m.extra["threshold"] == 0.5


# ---------------------------------------------------------------------------
# Length mismatch guard
# ---------------------------------------------------------------------------

def test_flag_label_length_mismatch_raises(policy):
    handle = StubHandle(primary_metric="f1_score", n=10)
    result = RunResult(
        flags=[True]*5, scores=[0.9]*5, explanation=[""]*5  # wrong length
    )
    with pytest.raises(AssertionError, match="flags"):
        evaluate(result, handle, policy)
```

---

### 4. `tests/test_playground.py` — update `StubPattern` to remove `primary_metric_value`

`StubPattern` currently sets `r.primary_metric_value = self._primary`.
Now that the field is removed from `RunResult`, that line will raise
`AttributeError`. Remove it. Also remove the `primary_metric` constructor
parameter — the pattern no longer has any concept of a primary metric.

The playground tests should continue to verify what they always verified:
that runs write to `runs.json`, the registry updates, status and tier
are separate fields. None of that changes. Only `StubPattern` needs updating.

**Replace `StubPattern` in `tests/test_playground.py` with this:**

```python
class StubPattern(PatternHandler):
    name    = "stub_pattern"
    version = "0.1"

    def run(self, handle):
        df = handle.eval_df()
        n  = len(df)
        return RunResult(
            flags       = [True]*3 + [False]*(n - 3),
            scores      = [0.9]*3  + [0.1]*(n - 3),
            explanation = ["high signal"]*3 + [""]*(n - 3),
        )
        # Note: no primary_metric_value — evaluator computes this

    def describe(self):
        return {"pattern": self.name}
```

Remove the `primary_metric=0.8` argument from all calls to `StubPattern()`
in the test file. Remove `test_second_run_status_discard_if_no_improvement`'s
`StubPattern(primary_metric=0.8)` argument.

---

## What does NOT change

These files are unmodified by this spec:

| File | Reason |
|------|--------|
| `datasets/base.py` | `DatasetMeta.primary_metric: str` already exists and is correct |
| `lab/playground.py` | calls `evaluate(result, handle, policy)` — signature unchanged |
| `lab/arena.py` | unchanged |
| `memory/registry.py` | unchanged |
| `runtime/config.py` | unchanged |
| `runtime/llm.py` | unchanged |
| `scoring_policy.yaml` | unchanged — `primary_metric: 0.40` weight still valid |
| `main.py` | unchanged |

---

## Downstream impact on Phase 2 fraud patterns

If Phase 2 (`use_cases/fraud/`) has already been built, three things need
updating there too. They are not in the engine — they are in the use-case.

**Remove from every fraud pattern's `run()` method:**
```python
# DELETE these lines wherever they appear:
result.primary_metric_value = compute_f1(flags, labels)
r.primary_metric_value = compute_f1(flags, labels)
```

**Remove `compute_f1` from `use_cases/fraud/patterns/__init__.py`** entirely.
The evaluator now computes F1. The pattern helper is redundant and confusing.

**The `_precision` and `_recall` helpers stay** — they feed `extra_metrics`
for logging, which is still valid. Example of the correct pattern after fix:

```python
def run(self, handle):
    df     = handle.eval_df()
    labels = handle.labels()
    n      = len(df)

    flags, scores, explanation = self._detect(df)

    result = RunResult(flags=flags, scores=scores, explanation=explanation)
    # extra_metrics = supplementary diagnostics for logging only
    result.extra_metrics = {
        "precision": _precision(flags, labels),
        "recall":    _recall(flags, labels),
        "n_flagged": sum(flags),
        "threshold": self.threshold,
    }
    # Do NOT set primary_metric_value — evaluator owns this
    return result
```

**Update `use_cases/fraud/tests/test_fraud_patterns.py`:**

The `TestPatternContract` parametrized test currently has:
```python
def test_primary_metric_in_range(self, pattern):
    result = pattern.detect(handle)
    assert 0.0 <= result.primary_metric_value <= 1.0
```
Delete this test. `primary_metric_value` no longer exists on `RunResult`.
Replace it with a test that confirms the field is gone:
```python
def test_runresult_has_no_primary_metric_value(self, pattern):
    handle = StubFraudHandle()
    result = pattern.detect(handle)
    assert not hasattr(result, "primary_metric_value"), \
        "Patterns must not set primary_metric_value — evaluator owns this"
```

---

## Deliverables checklist

- [ ] `patterns/base.py` — `primary_metric_value` removed from `RunResult`. Docstring updated.
- [ ] `evaluation/evaluator.py` — `compute_primary_metric()` added. `evaluate()` calls it using `handle.meta.primary_metric`. `EvalMetrics` has `primary_metric_name` field.
- [ ] `tests/test_evaluator.py` — fully rewritten per this spec.
- [ ] `tests/test_playground.py` — `StubPattern` updated, `primary_metric` arg removed.
- [ ] `uv run pytest tests/ -q` — zero failures.
- [ ] If Phase 2 is built: fraud patterns updated, `compute_f1` removed, `test_fraud_patterns.py` updated.
- [ ] Verify `datasets/base.py` is **unchanged** — `git diff datasets/base.py` must be empty.
- [ ] Verify `scoring_policy.yaml` is **unchanged**.
- [ ] Verify `lab/playground.py` is **unchanged**.

## What Claude Code must NOT do

- Do not add `primary_metric_value` back to `RunResult` under any other name.
  The pattern's output is `flags`, `scores`, `explanation`. Period.
- Do not let patterns call `compute_primary_metric()` or import from
  `evaluation/evaluator.py`. That import direction is wrong — evaluator
  depends on patterns, not the reverse.
- Do not add a fallback that reads `result.primary_metric_value` if the
  field still happens to exist. The old path must be completely gone.
- Do not change `datasets/base.py`. The `primary_metric: str` field is
  already there and already correct — it just wasn't being read.

---

## Why this matters for scoring calibration

After this fix, the weight debate becomes meaningful. Currently,
`primary_metric: 0.40` is a weight on an untrusted self-reported number.
After this fix, it is a weight on a number the engine computed itself,
consistently, using the same function, for every pattern on every dataset.

When the registry shows `rule_spike` at `confidence=0.55` and `ml_logistic`
at `confidence=0.63`, those numbers will mean the same thing:

```
60% — f1_score computed by evaluator from flags vs ground truth
25% — fraction of flagged rows with non-empty explanation
...
```

Patterns cannot inflate their score. The dataset controls what "good" means.
The evaluator enforces it. The registry reflects it.
