"""Registry tests — verifies status/tier separation."""

import json
import pytest
from pathlib import Path
from memory.registry import PatternRegistry, append_run


def _run(pattern, score, status, timestamp, domain="test", dataset_id="stub"):
    return {
        "pattern":      pattern,
        "domain":       domain,
        "version":      "0.1",
        "dataset_id":   dataset_id,
        "pattern_path": f"patterns/scratch/{pattern}.py",
        "config":       {},
        "score":        score,
        "status":       status,
        "tier":         "scratch",
        "commit":       "abc1234",
        "timestamp":    timestamp,
        "description":  f"{pattern} test run",
        "metrics":      {"primary_metric_value": score},
    }


def _write_runs(path: Path, runs: list[dict]) -> None:
    path.write_text(json.dumps(runs, indent=2))


def test_status_and_tier_are_separate(tmp_path):
    runs_path     = tmp_path / "runs.json"
    registry_path = tmp_path / "registry.json"
    _write_runs(runs_path, [
        _run("pattern_a", 0.72, "keep", 1000),
    ])
    reg = PatternRegistry.load(runs_path, registry_path)
    e = reg.get("pattern_a")
    assert e.last_status == "keep"     # run outcome
    assert e.tier == "working"         # promotion level (0.72 >= 0.65)


def test_discard_status_does_not_affect_tier(tmp_path):
    runs_path     = tmp_path / "runs.json"
    registry_path = tmp_path / "registry.json"
    _write_runs(runs_path, [
        _run("pattern_a", 0.72, "keep",    1000),
        _run("pattern_a", 0.60, "discard", 2000),
    ])
    reg = PatternRegistry.load(runs_path, registry_path)
    e = reg.get("pattern_a")
    assert e.last_status == "discard"
    assert e.tier == "working"
    assert e.confidence == pytest.approx(0.72)
    assert e.runs == 2


def test_scratch_tier_below_threshold(tmp_path):
    runs_path     = tmp_path / "runs.json"
    registry_path = tmp_path / "registry.json"
    _write_runs(runs_path, [
        _run("pattern_b", 0.50, "keep", 1000),
    ])
    reg = PatternRegistry.load(runs_path, registry_path)
    assert reg.get("pattern_b").tier == "scratch"


def test_stable_tier_preserved_across_rebuild(tmp_path):
    runs_path     = tmp_path / "runs.json"
    registry_path = tmp_path / "registry.json"
    _write_runs(runs_path, [
        _run("pattern_a", 0.82, "keep", 1000),
    ])
    registry_path.write_text(json.dumps({
        "pattern_a": {"tier": "stable", "confidence": 0.82}
    }))
    reg = PatternRegistry.load(runs_path, registry_path)
    assert reg.get("pattern_a").tier == "stable"


def test_promotion_candidate_requires_min_runs(tmp_path):
    runs_path     = tmp_path / "runs.json"
    registry_path = tmp_path / "registry.json"
    _write_runs(runs_path, [
        _run("pattern_a", 0.82, "keep", 1000),
    ])
    reg = PatternRegistry.load(runs_path, registry_path)
    assert reg.get("pattern_a").promotion_candidate is False


def test_promotion_candidate_after_min_runs(tmp_path):
    runs_path     = tmp_path / "runs.json"
    registry_path = tmp_path / "registry.json"
    runs = [_run("pattern_a", 0.82, "keep", i) for i in range(3)]
    _write_runs(runs_path, runs)
    reg = PatternRegistry.load(runs_path, registry_path)
    assert reg.get("pattern_a").promotion_candidate is True


def test_best_metrics_are_domain_agnostic(tmp_path):
    runs_path     = tmp_path / "runs.json"
    registry_path = tmp_path / "registry.json"
    run = _run("pattern_a", 0.72, "keep", 1000)
    run["metrics"] = {"primary_metric_value": 0.72, "custom_kpi": 0.88}
    _write_runs(runs_path, [run])
    reg = PatternRegistry.load(runs_path, registry_path)
    e = reg.get("pattern_a")
    assert "custom_kpi" in e.best_metrics


def test_append_run_creates_file(tmp_path):
    runs_path = tmp_path / "runs.json"
    append_run({"pattern": "p", "score": 0.5}, runs_path)
    assert runs_path.exists()
    data = json.loads(runs_path.read_text())
    assert len(data) == 1


def test_append_run_is_append_only(tmp_path):
    runs_path = tmp_path / "runs.json"
    append_run({"pattern": "p", "score": 0.5}, runs_path)
    append_run({"pattern": "p", "score": 0.6}, runs_path)
    data = json.loads(runs_path.read_text())
    assert len(data) == 2
    assert data[0]["score"] == pytest.approx(0.5)
