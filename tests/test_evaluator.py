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
