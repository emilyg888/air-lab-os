"""Registry tests for registry.json as the source of truth."""

import json
from pathlib import Path

import pytest

from core.registry import (
    PatternRegistry,
    append_run,
    apply_promotion,
    load_registry,
    update_registry,
)


def _write_registry(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _policy() -> dict:
    return {
        "promotion": {
            "working_threshold": 0.65,
            "stable_threshold": 0.78,
            "min_runs": 3,
        },
        "rules": {
            "stability_threshold": 0.05,
            "promotion_confidence_threshold": 0.7,
        },
    }


def test_registry_loads_from_saved_registry_json(tmp_path):
    registry_path = tmp_path / "registry.json"
    _write_registry(
        registry_path,
        {
            "velocity_v1": {
                "runs": 5,
                "scores": [0.62, 0.68, 0.71, 0.73, 0.74],
                "avg_score": 0.696,
                "last_score": 0.74,
                "confidence": 0.6701,
                "status": "silver",
                "is_stable": True,
                "last_updated": "1970-01-01T01:23:20",
            }
        },
    )

    runs_path = tmp_path / "runs.json"
    entry = PatternRegistry.load(runs_path=runs_path, registry_path=registry_path).get("velocity_v1")

    assert entry.runs == 5
    assert entry.scores == pytest.approx([0.62, 0.68, 0.71, 0.73, 0.74])
    assert entry.avg_score == pytest.approx(0.696)
    assert entry.last_score == pytest.approx(0.74)
    assert entry.confidence == pytest.approx(0.6701)
    assert entry.status == "silver"
    assert entry.is_stable is True
    assert entry.last_updated == "1970-01-01T01:23:20"


def test_registry_backfills_legacy_saved_entry(tmp_path):
    registry_path = tmp_path / "registry.json"
    _write_registry(
        registry_path,
        {
            "rule_spike": {
                "runs": 1,
                "confidence": 0.6162,
                "last_score": 0.6162,
                "status": "bronze",
                "is_stable": False,
                "last_updated": "2026-04-09T10:00:00",
            }
        },
    )

    runs_path = tmp_path / "runs.json"
    entry = PatternRegistry.load(runs_path=runs_path, registry_path=registry_path).get("rule_spike")

    assert entry.scores == [0.6162]
    assert entry.avg_score == pytest.approx(0.6162)
    assert entry.status == "bronze"


def test_update_registry_persists_decision_state(tmp_path):
    registry_path = tmp_path / "registry.json"
    policy = _policy()

    update_registry("pattern_a", 0.81, {}, policy, path=registry_path)
    update_registry("pattern_a", 0.82, {}, policy, path=registry_path)
    update_registry("pattern_a", 0.80, {}, policy, path=registry_path)

    runs_path = tmp_path / "runs.json"
    entry = PatternRegistry.load(runs_path=runs_path, registry_path=registry_path).get("pattern_a")

    assert entry.avg_score == pytest.approx(0.81)
    assert entry.is_stable is True
    assert entry.status == "bronze"
    assert entry.promotion_candidate == "silver"
    assert load_registry(registry_path)["pattern_a"]["scores"] == [0.81, 0.82, 0.8]


def test_apply_promotion_updates_status_and_clears_candidate(tmp_path):
    registry_path = tmp_path / "registry.json"
    policy = _policy()

    update_registry("pattern_a", 0.81, {}, policy, path=registry_path)
    update_registry("pattern_a", 0.82, {}, policy, path=registry_path)
    update_registry("pattern_a", 0.80, {}, policy, path=registry_path)

    apply_promotion("pattern_a", "silver", path=registry_path)

    runs_path = tmp_path / "runs.json"
    entry = PatternRegistry.load(runs_path=runs_path, registry_path=registry_path).get("pattern_a")

    assert entry.status == "silver"
    assert entry.promotion_candidate is None


def test_best_score_uses_peak_saved_score(tmp_path):
    registry_path = tmp_path / "registry.json"
    _write_registry(
        registry_path,
        {
            "pattern_a": {
                "runs": 2,
                "scores": [0.72, 0.60],
                "avg_score": 0.66,
                "last_score": 0.60,
                "confidence": 0.44,
                "status": "silver",
                "is_stable": False,
                "last_updated": "2026-04-09T10:00:00",
                "last_status": "discard",
            }
        },
    )

    runs_path = tmp_path / "runs.json"
    registry = PatternRegistry.load(runs_path=runs_path, registry_path=registry_path)
    entry = registry.get("pattern_a")

    assert registry.best_score("pattern_a") == pytest.approx(0.72)
    assert entry.status == "silver"
    assert entry.tier == "working"
    assert entry.last_status == "discard"


def test_save_writes_compact_registry_shape(tmp_path):
    registry_path = tmp_path / "registry.json"
    update_registry(
        "velocity_v1",
        0.62,
        {},
        _policy(),
        path=registry_path,
    )
    update_registry(
        "velocity_v1",
        0.68,
        {},
        _policy(),
        path=registry_path,
    )

    data = json.loads(registry_path.read_text())

    assert set(data["velocity_v1"]) == {
        "use_case",
        "pattern_name",
        "runs",
        "scores",
        "avg_score",
        "last_score",
        "confidence",
        "status",
        "is_stable",
        "promotion_candidate",
        "last_updated",
    }


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
