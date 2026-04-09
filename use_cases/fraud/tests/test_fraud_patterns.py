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
