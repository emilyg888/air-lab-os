"""
erg_consistency_spike — flags sessions where the rolling split stddev
over the trailing window exceeds a threshold. Erratic splits usually
accompany neuromuscular fatigue or dehydration.

Hyperparameters:
    stddev_max: rolling stddev threshold (seconds per 500m)
"""

from __future__ import annotations

from datasets.base import DatasetHandle
from patterns.base import PatternHandler, RunResult


class ErgConsistencySpike(PatternHandler):
    name    = "erg_consistency_spike"
    version = "0.1"

    def __init__(self, stddev_max: float = 10.0):
        self.stddev_max = float(stddev_max)

    def run(self, handle: DatasetHandle) -> RunResult:
        df = handle.eval_df()
        stddevs = df["consistency"].astype(float)

        flags: list[bool] = []
        scores: list[float] = []
        explanation: list[str] = []

        scale = max(self.stddev_max, 1.0)
        for value in stddevs:
            flagged = value > self.stddev_max
            flags.append(bool(flagged))
            scores.append(min(1.0, max(0.0, float(value) / (scale * 2))))
            explanation.append(
                f"rolling stddev={value:.1f} > {self.stddev_max:.1f}"
                if flagged else ""
            )

        result = RunResult(flags=flags, scores=scores, explanation=explanation)
        result.extra_metrics = {
            "stddev_max": self.stddev_max,
            "n_flagged":  int(sum(flags)),
        }
        return result

    def describe(self) -> dict:
        return {
            "pattern":    self.name,
            "version":    self.version,
            "stddev_max": self.stddev_max,
        }


def get_pattern(**config) -> ErgConsistencySpike:
    return ErgConsistencySpike(**config)
