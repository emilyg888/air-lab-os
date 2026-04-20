"""
erg_load_threshold — flags sessions where 7-day rolling training load
exceeds a threshold. Simple cumulative-stress signal for fatigue risk.

Hyperparameters the planner can vary:
    load_max_min: weekly minutes threshold (lower = more sensitive)
"""

from __future__ import annotations

from datasets.base import DatasetHandle
from patterns.base import PatternHandler, RunResult


class ErgLoadThreshold(PatternHandler):
    name    = "erg_load_threshold"
    version = "0.1"

    def __init__(self, load_max_min: float = 90.0):
        self.load_max_min = float(load_max_min)

    def run(self, handle: DatasetHandle) -> RunResult:
        df = handle.eval_df()
        load = df["weekly_load_min"].astype(float)

        flags: list[bool] = []
        scores: list[float] = []
        explanation: list[str] = []

        scale = max(self.load_max_min, 1.0)
        for value in load:
            flagged = value > self.load_max_min
            flags.append(bool(flagged))
            # Score saturates at 2x threshold for clean [0, 1] range.
            scores.append(min(1.0, max(0.0, float(value) / (scale * 2))))
            explanation.append(
                f"weekly_load={value:.1f}min > {self.load_max_min:.0f}min"
                if flagged else ""
            )

        result = RunResult(flags=flags, scores=scores, explanation=explanation)
        result.extra_metrics = {
            "load_max_min": self.load_max_min,
            "n_flagged":    int(sum(flags)),
        }
        return result

    def describe(self) -> dict:
        return {
            "pattern":      self.name,
            "version":      self.version,
            "load_max_min": self.load_max_min,
        }


def get_pattern(**config) -> ErgLoadThreshold:
    """Engine entry point. Accepts hyperparameter overrides from the planner."""
    return ErgLoadThreshold(**config)
