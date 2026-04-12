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
