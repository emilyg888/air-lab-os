from types import SimpleNamespace

from core.loop import run_lab
from core.mode import Mode


class StubPattern:
    def __init__(self, name: str):
        self.name = name


def _policy() -> dict:
    return {
        "promotion": {"working_threshold": 0.65},
        "rules": {"stability_gap_threshold": 0.2},
    }


def test_run_lab_executes_supplied_patterns_in_order(monkeypatch):
    run_calls = []
    update_calls = []

    monkeypatch.setattr("core.loop._load_dataset", lambda dataset_name: {"name": dataset_name})
    monkeypatch.setattr(
        "core.loop._run_experiment",
        lambda pattern, dataset: run_calls.append((pattern.name, dataset["name"])) or {},
    )
    monkeypatch.setattr(
        "core.loop.evaluate",
        lambda result, dataset, policy, mode: SimpleNamespace(score=0.9),
    )
    monkeypatch.setattr(
        "core.loop._update_registry",
        lambda pattern_name, score, metadata, policy: update_calls.append((pattern_name, score, metadata)),
    )

    results = run_lab(
        patterns=[StubPattern("first"), StubPattern("second")],
        explore_dataset="explore",
        validate_dataset="validate",
        policy=_policy(),
    )

    assert run_calls == [
        ("first", "explore"),
        ("first", "validate"),
        ("second", "explore"),
        ("second", "validate"),
    ]
    assert [result["pattern"] for result in results] == ["first", "second"]
    assert [call[0] for call in update_calls] == ["first", "second"]


def test_run_lab_does_not_prune_in_exploration_mode(monkeypatch):
    run_calls = []
    update_calls = []

    monkeypatch.setattr("core.loop._load_dataset", lambda dataset_name: {"name": dataset_name})
    monkeypatch.setattr(
        "core.loop._run_experiment",
        lambda pattern, dataset: run_calls.append((pattern.name, dataset["name"])) or {},
    )
    monkeypatch.setattr(
        "core.loop.evaluate",
        lambda result, dataset, policy, mode: SimpleNamespace(
            score=0.5 if dataset["name"] == "explore" else 0.9
        ),
    )
    monkeypatch.setattr(
        "core.loop._update_registry",
        lambda pattern_name, score, metadata, policy: update_calls.append((pattern_name, score)),
    )

    results = run_lab(
        patterns=[StubPattern("candidate")],
        explore_dataset="explore",
        validate_dataset="validate",
        policy=_policy(),
        mode=Mode.EXPLORATION,
    )

    assert run_calls == [("candidate", "explore"), ("candidate", "validate")]
    assert len(update_calls) == 1
    assert results == [
        {
            "pattern": "candidate",
            "explore_score": 0.5,
            "validation_score": 0.9,
            "stability_gap": 0.4,
            "is_stable": False,
        }
    ]


def test_run_lab_execution_mode_does_not_early_prune(monkeypatch):
    run_calls = []
    update_calls = []

    monkeypatch.setattr("core.loop._load_dataset", lambda dataset_name: {"name": dataset_name})
    monkeypatch.setattr(
        "core.loop._run_experiment",
        lambda pattern, dataset: run_calls.append((pattern.name, dataset["name"])) or {},
    )
    monkeypatch.setattr(
        "core.loop.evaluate",
        lambda result, dataset, policy, mode: SimpleNamespace(
            score=0.5 if dataset["name"] == "explore" else 0.9
        ),
    )
    monkeypatch.setattr(
        "core.loop._update_registry",
        lambda pattern_name, score, metadata, policy: update_calls.append((pattern_name, score, metadata)),
    )

    results = run_lab(
        patterns=[StubPattern("candidate")],
        explore_dataset="explore",
        validate_dataset="validate",
        policy=_policy(),
        mode=Mode.EXECUTION,
    )

    assert run_calls == [("candidate", "explore"), ("candidate", "validate")]
    assert len(update_calls) == 1
    assert update_calls[0][0] == "candidate"
    assert results == [
        {
            "pattern": "candidate",
            "explore_score": 0.5,
            "validation_score": 0.9,
            "stability_gap": 0.4,
            "is_stable": False,
        }
    ]


def test_run_lab_passes_mode_into_registry_metadata(monkeypatch):
    update_calls = []

    monkeypatch.setattr("core.loop._load_dataset", lambda dataset_name: {"name": dataset_name})
    monkeypatch.setattr("core.loop._run_experiment", lambda pattern, dataset: {})
    monkeypatch.setattr("core.loop.evaluate", lambda result, dataset, policy, mode: SimpleNamespace(score=0.9))
    monkeypatch.setattr(
        "core.loop._update_registry",
        lambda pattern_name, score, metadata, policy: update_calls.append(metadata),
    )

    run_lab(
        patterns=[StubPattern("candidate")],
        explore_dataset="explore",
        validate_dataset="validate",
        policy=_policy(),
        mode=Mode.EXECUTION,
    )

    assert update_calls[0]["mode"] == "execution"
