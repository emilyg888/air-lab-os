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
