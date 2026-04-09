import json

from core.promotion import promote_patterns
from core.registry import load_registry, update_registry


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


def test_promote_patterns_dry_run_only_plans_moves(tmp_path):
    registry_path = tmp_path / "registry.json"
    base_dir = tmp_path / "patterns"
    source = base_dir / "scratch" / "pattern_a.py"
    source.parent.mkdir(parents=True)
    source.write_text("# test pattern\n")

    update_registry("pattern_a", 0.81, {}, _policy(), path=registry_path)
    update_registry("pattern_a", 0.82, {}, _policy(), path=registry_path)
    update_registry("pattern_a", 0.80, {}, _policy(), path=registry_path)

    actions = promote_patterns(dry_run=True, registry_path=registry_path, base_dir=base_dir)

    assert len(actions) == 1
    assert actions[0].source == source
    assert actions[0].destination == base_dir / "working" / "pattern_a.py"
    assert source.exists()
    assert not actions[0].destination.exists()
    assert load_registry(registry_path)["pattern_a"]["status"] == "bronze"


def test_promote_patterns_apply_updates_registry_without_moving_files_by_default(tmp_path):
    registry_path = tmp_path / "registry.json"
    base_dir = tmp_path / "patterns"
    source = base_dir / "scratch" / "pattern_a.py"
    source.parent.mkdir(parents=True)
    source.write_text("# test pattern\n")

    update_registry("pattern_a", 0.81, {}, _policy(), path=registry_path)
    update_registry("pattern_a", 0.82, {}, _policy(), path=registry_path)
    update_registry("pattern_a", 0.80, {}, _policy(), path=registry_path)

    actions = promote_patterns(dry_run=False, registry_path=registry_path, base_dir=base_dir)
    registry = load_registry(registry_path)

    assert len(actions) == 1
    assert actions[0].applied is True
    assert actions[0].filesystem_reflected is False
    assert source.exists()
    assert not (base_dir / "working" / "pattern_a.py").exists()
    assert registry["pattern_a"]["status"] == "silver"
    assert registry["pattern_a"]["promotion_candidate"] is None


def test_promote_patterns_can_reflect_filesystem_when_requested(tmp_path):
    registry_path = tmp_path / "registry.json"
    base_dir = tmp_path / "patterns"
    source = base_dir / "scratch" / "pattern_a.py"
    source.parent.mkdir(parents=True)
    source.write_text("# test pattern\n")

    update_registry("pattern_a", 0.81, {}, _policy(), path=registry_path)
    update_registry("pattern_a", 0.82, {}, _policy(), path=registry_path)
    update_registry("pattern_a", 0.80, {}, _policy(), path=registry_path)

    actions = promote_patterns(
        dry_run=False,
        reflect_filesystem=True,
        registry_path=registry_path,
        base_dir=base_dir,
    )

    assert len(actions) == 1
    assert actions[0].applied is True
    assert actions[0].filesystem_reflected is True
    assert not source.exists()
    assert (base_dir / "working" / "pattern_a.py").exists()


def test_promote_patterns_skips_missing_source_files_when_reflecting_filesystem(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "pattern_a": {
                    "runs": 3,
                    "scores": [0.81, 0.82, 0.80],
                    "avg_score": 0.81,
                    "last_score": 0.80,
                    "confidence": 0.81,
                    "status": "bronze",
                    "is_stable": True,
                    "promotion_candidate": "silver",
                    "last_updated": "2026-04-09T00:00:00",
                }
            }
        )
    )

    actions = promote_patterns(
        dry_run=False,
        reflect_filesystem=True,
        registry_path=registry_path,
        base_dir=tmp_path / "patterns",
    )

    assert len(actions) == 1
    assert actions[0].applied is False
    assert actions[0].skipped_reason == "source file is missing"
    assert load_registry(registry_path)["pattern_a"]["status"] == "bronze"
