"""Evaluator tests — no domain imports, pure engine logic."""

import pytest
from evaluation.evaluator import evaluate, load_policy, EvalMetrics
from patterns.base import RunResult
from datasets.base import DatasetHandle, DatasetMeta
import pandas as pd


class StubHandle(DatasetHandle):
    """Minimal DatasetHandle for testing."""
    def __init__(self, n=10, n_positive=3):
        self._n = n
        self._n_pos = n_positive

    @property
    def meta(self) -> DatasetMeta:
        return DatasetMeta(
            name="stub", domain="test", tier="bronze", version="0.1",
            label_column="label", primary_metric="f1_score",
            row_count=self._n,
        )

    def eval_df(self) -> pd.DataFrame:
        return pd.DataFrame({"label": [True]*self._n_pos + [False]*(self._n-self._n_pos)})

    def train_df(self) -> pd.DataFrame:
        return self.eval_df()

    def labels(self) -> list[bool]:
        return [True]*self._n_pos + [False]*(self._n-self._n_pos)


@pytest.fixture
def policy():
    return load_policy()


@pytest.fixture
def handle():
    return StubHandle()


def _result(primary=1.0, with_expl=True, latency_ms=10.0, n=10, n_pos=3):
    flags = [True]*n_pos + [False]*(n-n_pos)
    scores = [0.9]*n_pos + [0.1]*(n-n_pos)
    expl = (["high signal"]*n_pos + [""]*( n-n_pos)) if with_expl else [""]*n
    r = RunResult(flags=flags, scores=scores, explanation=expl, latency_ms=latency_ms)
    r.primary_metric_value = primary
    return r


def test_score_in_range(policy, handle):
    m = evaluate(_result(), handle, policy)
    assert 0.0 <= m.score <= 1.0


def test_perfect_primary_metric(policy, handle):
    m = evaluate(_result(primary=1.0), handle, policy)
    assert m.primary_metric_value == pytest.approx(1.0)


def test_zero_primary_metric(policy, handle):
    m = evaluate(_result(primary=0.0), handle, policy)
    assert m.primary_metric_value == pytest.approx(0.0)


def test_full_explainability(policy, handle):
    m = evaluate(_result(with_expl=True), handle, policy)
    assert m.explainability_score == pytest.approx(1.0)


def test_zero_explainability(policy, handle):
    m = evaluate(_result(with_expl=False), handle, policy)
    assert m.explainability_score == pytest.approx(0.0)


def test_latency_penalised(policy, handle):
    fast = evaluate(_result(latency_ms=0.0), handle, policy)
    slow = evaluate(_result(latency_ms=500.0), handle, policy)
    assert fast.latency_score > slow.latency_score
    assert slow.latency_score == pytest.approx(0.0)


def test_no_flags_explainability_is_one(policy, handle):
    r = RunResult(flags=[False]*10, scores=[0.0]*10, explanation=[""]*10)
    r.primary_metric_value = 0.0
    m = evaluate(r, handle, policy)
    assert m.explainability_score == pytest.approx(1.0)


def test_weights_sum_to_one(policy):
    total = (
        policy.primary_metric_weight
        + policy.explainability_weight
        + policy.latency_weight
        + policy.cost_weight
    )
    assert abs(total - 1.0) < 1e-6
