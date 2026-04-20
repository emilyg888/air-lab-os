from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class DetectionResult:
    """
    Output contract for every detector.

    All three lists must be the same length as the input DataFrame.
    flags       — True if transaction is flagged as fraud
    scores      — float in [0.0, 1.0], fraud probability or confidence
    explanation — human-readable reason string, or "" if none
    latency_ms  — wall-clock milliseconds for the full run() call
    cost_per_1k — estimated USD cost per 1000 transactions (0.0 for local models)
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
        assert all(0.0 <= s <= 1.0 for s in self.scores), "scores must be in [0, 1]"


class Detector(ABC):
    """
    Abstract base for all fraud detectors.

    Subclasses implement run(). The base class wraps run() to
    record wall-clock latency automatically.

    Usage:
        detector = VelocityDetector(window_seconds=3600, threshold=5)
        result = detector.detect(df)
    """

    name: str = "base"

    def detect(self, data: pd.DataFrame) -> DetectionResult:
        """
        Public entry point. Times the call and injects latency_ms.
        Do not override — override run() instead.
        """
        import time
        t0 = time.perf_counter()
        result = self.run(data)
        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    @abstractmethod
    def run(self, data: pd.DataFrame) -> DetectionResult:
        """
        Implement fraud detection logic here.

        Args:
            data: DataFrame with at least these columns:
                  transaction_id (str), amount (float),
                  user_id (str), timestamp (int, unix seconds),
                  merchant_id (str)

        Returns:
            DetectionResult with flags, scores, explanation lists
            the same length as len(data).
            Do NOT set latency_ms — the base class sets it.
        """
        ...

    def describe(self) -> dict:
        """Return config dict for logging. Override to add hyperparams."""
        return {"detector": self.name}
