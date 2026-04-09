"""Playground tests — mocks PatternHandler and DatasetHandle."""

import json
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock

from datasets.base import DatasetHandle, DatasetMeta
from patterns.base import PatternHandler, RunResult
from lab.playground import run_experiment


class StubHandle(DatasetHandle):
    @property
    def meta(self):
        return DatasetMeta(
            name="stub", domain="test", tier="bronze", version="0.1",
            label_column="label", primary_metric="f1_score", row_count=10,
        )
    def eval_df(self):
        return pd.DataFrame({"label": [True]*3 + [False]*7})
    def train_df(self):
        return self.eval_df()
    def labels(self):
        return [True]*3 + [False]*7


class StubPattern(PatternHandler):
    name    = "stub_pattern"
    version = "0.1"

    def run(self, handle):
        df = handle.eval_df()
        n  = len(df)
        return RunResult(
            flags       = [True]*3 + [False]*(n - 3),
            scores      = [0.9]*3  + [0.1]*(n - 3),
            explanation = ["high signal"]*3 + [""]*(n - 3),
        )
        # Note: no primary_metric_value — evaluator computes this

    def describe(self):
        return {"pattern": self.name}


class CrashPattern(PatternHandler):
    name    = "crash_pattern"
    version = "0.1"

    def run(self, handle):
        raise RuntimeError("intentional crash")

    def describe(self):
        return {"pattern": self.name}


@pytest.fixture
def tmp_paths(tmp_path):
    return {
        "runs":     tmp_path / "runs.json",
        "registry": tmp_path / "registry.json",
    }


def test_successful_run_writes_to_runs_json(tmp_paths):
    run_experiment(
        StubPattern(), StubHandle(),
        runs_path=tmp_paths["runs"],
        registry_path=tmp_paths["registry"],
    )
    assert tmp_paths["runs"].exists()
    data = json.loads(tmp_paths["runs"].read_text())
    assert len(data) == 1
    assert data[0]["pattern"] == "stub_pattern"
    assert data[0]["status"] in ("keep", "discard")


def test_crash_is_logged_not_raised(tmp_paths):
    result = run_experiment(
        CrashPattern(), StubHandle(),
        runs_path=tmp_paths["runs"],
        registry_path=tmp_paths["registry"],
    )
    assert result.status == "crash"
    assert tmp_paths["runs"].exists()


def test_result_has_separate_status_and_tier(tmp_paths):
    result = run_experiment(
        StubPattern(), StubHandle(),
        runs_path=tmp_paths["runs"],
        registry_path=tmp_paths["registry"],
    )
    assert result.status in ("keep", "discard", "crash")
    assert result.tier in ("scratch", "working", "stable")


def test_registry_updated_after_run(tmp_paths):
    run_experiment(
        StubPattern(), StubHandle(),
        runs_path=tmp_paths["runs"],
        registry_path=tmp_paths["registry"],
    )
    assert tmp_paths["registry"].exists()
    data = json.loads(tmp_paths["registry"].read_text())
    assert "stub_pattern" in data


def test_second_run_status_discard_if_no_improvement(tmp_paths):
    run_experiment(
        StubPattern(), StubHandle(),
        runs_path=tmp_paths["runs"],
        registry_path=tmp_paths["registry"],
    )
    r2 = run_experiment(
        StubPattern(), StubHandle(),
        runs_path=tmp_paths["runs"],
        registry_path=tmp_paths["registry"],
    )
    assert r2.status == "discard"
