"""
erg_split_regression — flags sessions where split pace regressed vs
the prior session by more than `delta_max_sec` seconds per 500m.

A positive delta = slower rowing. Large deltas on top of the athlete's
recent rolling mean are a classic fatigue tell.

Hyperparameters:
    delta_max_sec: per-session regression threshold in seconds per 500m
"""

from __future__ import annotations

from datasets.base import DatasetHandle
from patterns.base import PatternHandler, RunResult


class ErgSplitRegression(PatternHandler):
    name    = "erg_split_regression"
    version = "0.1"

    def __init__(self, delta_max_sec: float = 2.0):
        self.delta_max_sec = float(delta_max_sec)

    def run(self, handle: DatasetHandle) -> RunResult:
        df = handle.eval_df()
        deltas = df["delta"].astype(float)

        flags: list[bool] = []
        scores: list[float] = []
        explanation: list[str] = []

        scale = max(self.delta_max_sec, 0.5)
        for value in deltas:
            flagged = value > self.delta_max_sec
            flags.append(bool(flagged))
            scores.append(min(1.0, max(0.0, float(value) / (scale * 4))))
            explanation.append(
                f"split regressed by {value:.1f}s/500m (>{self.delta_max_sec:.1f})"
                if flagged else ""
            )

        result = RunResult(flags=flags, scores=scores, explanation=explanation)
        result.extra_metrics = {
            "delta_max_sec": self.delta_max_sec,
            "n_flagged":     int(sum(flags)),
        }
        return result

    def describe(self) -> dict:
        return {
            "pattern":       self.name,
            "version":       self.version,
            "delta_max_sec": self.delta_max_sec,
        }


def get_pattern(**config) -> ErgSplitRegression:
    return ErgSplitRegression(**config)
