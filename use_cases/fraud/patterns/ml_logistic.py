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
            "precision":  float(_precision(flags, labels)),
            "recall":     float(_recall(flags, labels)),
            "threshold":  self.threshold,
            "n_flagged":  int(sum(flags)),
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
