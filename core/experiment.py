"""Thin execution wrapper for the core loop."""

from __future__ import annotations

from datasets.base import DatasetHandle
from patterns.base import PatternHandler, RunResult


def run_experiment(pattern: PatternHandler, handle: DatasetHandle) -> RunResult:
    return pattern.detect(handle)
