"""Arena tests."""

import pytest
import pandas as pd
from pathlib import Path

from datasets.base import DatasetHandle, DatasetMeta
from patterns.base import PatternHandler, RunResult
from lab.arena import compare_patterns


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


def _make_pattern(name, score):
    class P(PatternHandler):
        def run(self, handle):
            df = handle.eval_df()
            n  = len(df)
            r  = RunResult(
                flags=[True]*3+[False]*7,
                scores=[score]*3+[0.1]*7,
                explanation=["sig"]*3+[""]*7,
            )
            r.primary_metric_value = score
            return r
        def describe(self):
            return {"pattern": self.name}
    P.name    = name
    P.version = "0.1"
    return P()


@pytest.fixture
def tmp_paths(tmp_path):
    return {"runs": tmp_path/"runs.json", "registry": tmp_path/"registry.json"}


def test_arena_ranks_by_score(tmp_paths):
    patterns = [
        _make_pattern("low",  0.5),
        _make_pattern("high", 0.9),
        _make_pattern("mid",  0.7),
    ]
    arena = compare_patterns(patterns, StubHandle(),
                             runs_path=tmp_paths["runs"],
                             registry_path=tmp_paths["registry"])
    scores = [r.score for r in arena.rankings]
    assert scores == sorted(scores, reverse=True)


def test_arena_winner_is_highest(tmp_paths):
    patterns = [_make_pattern("a", 0.6), _make_pattern("b", 0.85)]
    arena = compare_patterns(patterns, StubHandle(),
                             runs_path=tmp_paths["runs"],
                             registry_path=tmp_paths["registry"])
    assert arena.winner.pattern == "b"


def test_arena_logs_all_patterns(tmp_paths):
    import json
    patterns = [_make_pattern(f"p{i}", 0.5+i*0.1) for i in range(3)]
    compare_patterns(patterns, StubHandle(),
                     runs_path=tmp_paths["runs"],
                     registry_path=tmp_paths["registry"])
    runs = json.loads(tmp_paths["runs"].read_text())
    assert len(runs) == 3
