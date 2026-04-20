"""
runtime/evaluation/evaluator.py — FR-01
---------------------------------------
Multi-judge aggregator.

Replaces the single-call Evaluator with a per-dimension judge architecture.
Each configured dimension gets its own judge instance making one focused
LLM call. The composite is a weighted average of all judge scores.
"""

from __future__ import annotations

import logging

from runtime.config import RuntimeConfig
from runtime.evaluation.judges import (
    AlignmentJudge,
    BaseJudge,
    CompletenessJudge,
    CorrectnessJudge,
    GovernanceJudge,
)
from runtime.registry import PatternRegistry

log = logging.getLogger(__name__)


# Judge registry — dimension name → judge class
_JUDGE_REGISTRY: dict[str, type[BaseJudge]] = {
    "correctness":  CorrectnessJudge,
    "governance":   GovernanceJudge,
    "alignment":    AlignmentJudge,
    "completeness": CompletenessJudge,
}


class Evaluator:
    """
    Multi-judge aggregator. Each configured dimension gets its own judge
    instance. Composite score = weighted average of all judge scores.
    """

    def __init__(self, registry: PatternRegistry, config: RuntimeConfig) -> None:
        self._registry = registry
        self._config   = config
        self._judges: list[tuple[BaseJudge, float]] = self._build_judges()

    # ----------------------------------------------------------------------
    # Build
    # ----------------------------------------------------------------------

    def _build_judges(self) -> list[tuple[BaseJudge, float]]:
        """
        Instantiate one judge per configured dimension.
        Weight comes from config.evaluation_dimensions; falls back to the
        judge's default_weight if not in config.
        """
        configured = {d.name: d.weight for d in self._config.evaluation_dimensions}
        judges: list[tuple[BaseJudge, float]] = []

        if configured:
            # Build only the dimensions present in config (preserving order).
            for d in self._config.evaluation_dimensions:
                judge_cls = _JUDGE_REGISTRY.get(d.name)
                if judge_cls is None:
                    log.warning("evaluator: no judge registered for dimension %r", d.name)
                    continue
                judges.append((judge_cls(self._config), d.weight))
        else:
            # Fall back to all built-in judges with default weights.
            for dim_name, judge_cls in _JUDGE_REGISTRY.items():
                judges.append((judge_cls(self._config), judge_cls.default_weight))

        return judges

    # ----------------------------------------------------------------------
    # Evaluate
    # ----------------------------------------------------------------------

    def evaluate(self, result: dict, mode: str) -> dict:
        judge_results: list[dict] = []
        for judge, weight in self._judges:
            jr = judge.score(result, mode)
            jr["weight"] = weight
            judge_results.append(jr)

        total_w = sum(j["weight"] for j in judge_results)
        composite = (
            sum(j["score"] * j["weight"] for j in judge_results) / total_w
            if total_w
            else 0.0
        )
        rationale = " | ".join(
            f"{j['dimension']}={j['score']:.2f}: {j['rationale']}"
            for j in judge_results
        )

        output: dict = {
            "score":          round(composite, 4),
            "rationale":      rationale,
            "_dimensions":    [j["dimension"] for j in judge_results],
            "_weights":       {j["dimension"]: j["weight"] for j in judge_results},
            "_judge_results": judge_results,
        }
        for jr in judge_results:
            output[jr["dimension"]] = jr["score"]

        log.info("evaluator: composite=%.3f judges=%d", composite, len(judge_results))
        return output
