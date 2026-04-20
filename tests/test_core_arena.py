from core.arena import Mode, get_gold_patterns, rank_patterns, select_candidates, select_exploration_candidates


def test_rank_patterns_orders_by_rank_score(monkeypatch):
    monkeypatch.setattr(
        "core.arena.ranking.load_registry",
        lambda: {
            "alpha": {
                "avg_score": 0.8,
                "confidence": 0.8,
                "is_stable": True,
                "status": "silver",
                "runs": 3,
            },
            "beta": {
                "avg_score": 0.9,
                "confidence": 0.4,
                "is_stable": False,
                "status": "silver",
                "runs": 2,
            },
        },
    )

    ranked = rank_patterns()

    assert [entry["pattern"] for entry in ranked] == ["alpha", "beta"]


def test_select_exploration_candidates_skips_gold_and_prefers_underexplored(monkeypatch):
    monkeypatch.setattr(
        "core.arena.exploration.load_registry",
        lambda: {
            "gold": {"status": "gold", "runs": 4, "avg_score": 0.95},
            "fresh": {"status": "bronze", "runs": 0, "avg_score": 0.3},
            "promising": {"status": "silver", "runs": 1, "avg_score": 0.8},
            "mature": {"status": "silver", "runs": 4, "avg_score": 0.9},
        },
    )

    selected = select_exploration_candidates(top_n=3)

    assert [entry["pattern"] for entry in selected] == ["fresh", "promising", "mature"]
    assert all(entry["status"] != "gold" for entry in selected)


def test_select_exploration_candidates_respects_max_runs_budget(monkeypatch):
    monkeypatch.setattr(
        "core.arena.exploration.load_registry",
        lambda: {
            "fresh": {"status": "bronze", "runs": 0, "avg_score": 0.3},
            "budget_hit": {"status": "silver", "runs": 5, "avg_score": 0.95},
            "over_budget": {"status": "silver", "runs": 6, "avg_score": 0.99},
        },
    )

    selected = select_exploration_candidates(top_n=3)

    assert [entry["pattern"] for entry in selected] == ["fresh"]


def test_get_gold_patterns_returns_only_gold_sorted_by_avg_score(monkeypatch):
    monkeypatch.setattr(
        "core.arena.execution.load_registry",
        lambda: {
            "silver": {"status": "silver", "avg_score": 0.8, "confidence": 0.6, "runs": 2},
            "gold_low": {"status": "gold", "avg_score": 0.81, "confidence": 0.8, "runs": 4},
            "gold_high": {"status": "gold", "avg_score": 0.92, "confidence": 0.9, "runs": 5},
        },
    )

    gold = get_gold_patterns()

    assert [entry["pattern"] for entry in gold] == ["gold_high", "gold_low"]


def test_select_candidates_dispatches_by_mode(monkeypatch):
    monkeypatch.setattr(
        "core.arena.select_exploration_candidates",
        lambda top_n=3: [{"pattern": "explore"}],
    )
    monkeypatch.setattr(
        "core.arena.select_execution_candidates",
        lambda: [{"pattern": "execute"}],
    )

    assert select_candidates(Mode.EXPLORATION) == [{"pattern": "explore"}]
    assert select_candidates(Mode.EXECUTION) == [{"pattern": "execute"}]
