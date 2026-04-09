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


def _make_pattern(name, n_fp):
    """
    Build a pattern with 3 true positives and `n_fp` false positives.
    Higher n_fp → lower precision → lower F1 → lower composite score.
    """
    class P(PatternHandler):
        def run(self, handle):
            # 3 true positives + n_fp false positives from the negative tail
            flags = [True]*3 + [True]*n_fp + [False]*(7 - n_fp)
            return RunResult(
                flags=flags,
                scores=[0.9 if f else 0.1 for f in flags],
                explanation=["sig" if f else "" for f in flags],
            )
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
        _make_pattern("low",  n_fp=4),   # worst F1
        _make_pattern("high", n_fp=0),   # perfect F1
        _make_pattern("mid",  n_fp=2),   # middle F1
    ]
    arena = compare_patterns(patterns, StubHandle(),
                             runs_path=tmp_paths["runs"],
                             registry_path=tmp_paths["registry"])
    scores = [r.score for r in arena.rankings]
    assert scores == sorted(scores, reverse=True)


def test_arena_winner_is_highest(tmp_paths):
    patterns = [_make_pattern("a", n_fp=3), _make_pattern("b", n_fp=0)]
    arena = compare_patterns(patterns, StubHandle(),
                             runs_path=tmp_paths["runs"],
                             registry_path=tmp_paths["registry"])
    assert arena.winner.pattern == "b"


def test_arena_logs_all_patterns(tmp_paths):
    import json
    patterns = [_make_pattern(f"p{i}", n_fp=i) for i in range(3)]
    compare_patterns(patterns, StubHandle(),
                     runs_path=tmp_paths["runs"],
                     registry_path=tmp_paths["registry"])
    runs = json.loads(tmp_paths["runs"].read_text())
    assert len(runs) == 3
