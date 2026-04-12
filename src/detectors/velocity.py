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
