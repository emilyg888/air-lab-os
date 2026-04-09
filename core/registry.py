"""Unified registry layer for derived pattern memory."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional


RUNS_PATH = Path("memory/runs.json")
REGISTRY_PATH = Path("registry.json")

STATUS_TO_TIER = {
    "bronze": "scratch",
    "silver": "working",
    "gold": "stable",
}
TIER_TO_STATUS = {value: key for key, value in STATUS_TO_TIER.items()}
STABILITY_VARIANCE_THRESHOLD = 0.0025


def load_json(path: Path, default: Any) -> Any:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, ValueError):
            return default
    return default


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _iso_timestamp(unix_seconds: int) -> str:
    if unix_seconds <= 0:
        return ""
    dt = datetime.fromtimestamp(unix_seconds, tz=timezone.utc).replace(tzinfo=None)
    return dt.isoformat(timespec="seconds")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


def compute_confidence(avg_score: float, runs: int, variance: float, min_runs: int) -> float:
    sample_factor = min(1.0, runs / max(min_runs, 1))
    stability_factor = max(0.0, 1.0 - (variance * 20))
    return round(avg_score * sample_factor * stability_factor, 4)


def check_stability(scores: list[float], threshold: float = STABILITY_VARIANCE_THRESHOLD) -> bool:
    return len(scores) >= 3 and _variance(scores) <= threshold


def assign_status(
    avg_score: float,
    is_stable: bool,
    working_threshold: float,
    stable_threshold: float,
) -> str:
    if avg_score >= stable_threshold and is_stable:
        return "gold"
    if avg_score >= working_threshold:
        return "silver"
    return "bronze"


@dataclass
class RegistryEntry:
    pattern: str
    runs: int = 0
    scores: list[float] = field(default_factory=list)
    avg_score: float = 0.0
    last_score: float = 0.0
    confidence: float = 0.0
    status: str = "bronze"
    is_stable: bool = False
    last_updated: str = ""

    # Runtime-only compatibility fields.
    domain: str = ""
    version: str = "0.1"
    pattern_path: str = ""
    last_run_status: str = ""
    last_commit: str = ""
    last_dataset: str = ""
    best_metrics: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    @property
    def tier(self) -> str:
        return STATUS_TO_TIER.get(self.status, "scratch")

    @property
    def last_status(self) -> str:
        return self.last_run_status

    @property
    def promotion_candidate(self) -> bool:
        return (
            self.avg_score >= PatternRegistry.STABLE_THRESHOLD
            and self.runs >= PatternRegistry.MIN_RUNS
            and not self.is_stable
        )

    @property
    def max_score(self) -> float:
        return max(self.scores, default=0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "runs": self.runs,
            "scores": [round(score, 4) for score in self.scores],
            "avg_score": round(self.avg_score, 4),
            "last_score": round(self.last_score, 4),
            "confidence": round(self.confidence, 4),
            "status": self.status,
            "is_stable": self.is_stable,
            "last_updated": self.last_updated,
        }


class PatternRegistry:
    """Registry rebuilt from run history and saved as compact derived state."""

    WORKING_THRESHOLD: float = 0.65
    STABLE_THRESHOLD: float = 0.78
    MIN_RUNS: int = 3

    def __init__(self, entries: dict[str, RegistryEntry]):
        self._entries = entries

    @classmethod
    def load(
        cls,
        runs_path: Path = RUNS_PATH,
        registry_path: Path = REGISTRY_PATH,
    ) -> "PatternRegistry":
        del runs_path
        cls._load_policy_thresholds()

        entries: dict[str, RegistryEntry] = {}
        raw_registry = load_json(registry_path, {})
        for pattern, data in raw_registry.items():
            if not isinstance(data, dict):
                continue
            entries[pattern] = _entry_from_saved(pattern, data)

        return cls(entries)

    @classmethod
    def _load_policy_thresholds(cls) -> None:
        try:
            from runtime.config import load_policy

            policy = load_policy()
            cls.WORKING_THRESHOLD = policy.working_threshold
            cls.STABLE_THRESHOLD = policy.stable_threshold
            cls.MIN_RUNS = policy.min_runs
        except Exception:
            pass

    def get(self, pattern: str) -> Optional[RegistryEntry]:
        return self._entries.get(pattern)

    def all(self) -> list[RegistryEntry]:
        return sorted(self._entries.values(), key=lambda entry: entry.confidence, reverse=True)

    def by_tier(self, tier: str) -> list[RegistryEntry]:
        target_status = TIER_TO_STATUS.get(tier, tier)
        return [entry for entry in self._entries.values() if entry.status == target_status]

    def best_score(self, pattern: str) -> Optional[float]:
        entry = self._entries.get(pattern)
        return entry.max_score if entry else None

    def promotion_candidates(self) -> list[RegistryEntry]:
        return [entry for entry in self._entries.values() if entry.promotion_candidate]

    def save(self, path: Path = REGISTRY_PATH) -> None:
        save_registry({pattern: entry.to_dict() for pattern, entry in self._entries.items()}, path)


def _refresh_entry(entry: RegistryEntry, registry_cls: type[PatternRegistry]) -> None:
    entry.avg_score = _mean(entry.scores)
    variance = _variance(entry.scores)
    entry.confidence = compute_confidence(
        avg_score=entry.avg_score,
        runs=entry.runs,
        variance=variance,
        min_runs=registry_cls.MIN_RUNS,
    )
    entry.is_stable = check_stability(entry.scores)
    entry.status = assign_status(
        avg_score=entry.avg_score,
        is_stable=entry.is_stable,
        working_threshold=registry_cls.WORKING_THRESHOLD,
        stable_threshold=registry_cls.STABLE_THRESHOLD,
    )


def _entry_from_saved(pattern: str, data: dict[str, Any]) -> RegistryEntry:
    runs = max(int(data.get("runs", 0)), 0)
    scores = data.get("scores")
    if not isinstance(scores, list):
        seed_score = float(data.get("last_score", data.get("avg_score", data.get("confidence", 0.0))))
        scores = [round(seed_score, 4)] * runs if runs > 0 else []

    entry = RegistryEntry(
        pattern=pattern,
        runs=runs,
        scores=[round(float(score), 4) for score in scores],
        avg_score=float(data.get("avg_score", _mean(scores))),
        last_score=float(data.get("last_score", scores[-1] if scores else 0.0)),
        confidence=float(data.get("confidence", 0.0)),
        status=data.get("status", "bronze"),
        is_stable=bool(data.get("is_stable", False)),
        last_updated=str(data.get("last_updated", "")),
        domain=str(data.get("domain", "")),
        version=str(data.get("version", "0.1")),
        pattern_path=str(data.get("pattern_path", "")),
        last_run_status=str(data.get("last_status", data.get("last_run_status", ""))),
        last_commit=str(data.get("last_commit", "")),
        last_dataset=str(data.get("last_dataset", "")),
        best_metrics=data.get("best_metrics", {}) if isinstance(data.get("best_metrics"), dict) else {},
        description=str(data.get("description", "")),
    )
    if not entry.avg_score:
        entry.avg_score = _mean(entry.scores)
    if not entry.last_score and entry.scores:
        entry.last_score = entry.scores[-1]
    return entry


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    raw = load_json(path, {})
    if not isinstance(raw, dict):
        return {}
    return {
        pattern: _entry_from_saved(pattern, data).to_dict()
        for pattern, data in raw.items()
        if isinstance(data, dict)
    }


def save_registry(registry: dict[str, Any], path: Path = REGISTRY_PATH) -> None:
    save_json(path, registry)


def get_patterns_by_status(status: str, path: Path = REGISTRY_PATH) -> dict[str, Any]:
    registry = load_registry(path)
    return {name: entry for name, entry in registry.items() if entry.get("status") == status}


def get_top_patterns(n: int = 3, path: Path = REGISTRY_PATH) -> list[tuple[str, dict[str, Any]]]:
    registry = load_registry(path)
    return sorted(
        registry.items(),
        key=lambda item: item[1].get("avg_score", 0.0),
        reverse=True,
    )[:n]


def get_gold_patterns(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    return get_patterns_by_status("gold", path)


def update_registry(
    pattern_name: str,
    score: float,
    metadata: dict[str, Any],
    policy: Mapping[str, Any] | Any,
    path: Path = REGISTRY_PATH,
) -> dict[str, Any]:
    registry = load_registry(path)
    entry = registry.get(pattern_name, {"runs": 0, "scores": []})

    if "scores" not in entry or not isinstance(entry["scores"], list):
        prior_runs = max(int(entry.get("runs", 0)), 0)
        seed_score = float(
            entry.get("last_score", entry.get("avg_score", entry.get("confidence", score)))
        )
        entry["scores"] = [round(seed_score, 4)] * prior_runs

    entry["runs"] += 1
    entry["scores"].append(round(float(score), 4))
    entry["last_score"] = round(float(score), 4)

    avg_score = _mean(entry["scores"])
    variance = _variance(entry["scores"])
    if isinstance(policy, Mapping):
        min_runs = policy.get("promotion", {}).get("min_runs", PatternRegistry.MIN_RUNS)
        working_threshold = policy.get("promotion", {}).get(
            "working_threshold", PatternRegistry.WORKING_THRESHOLD
        )
        stable_threshold = policy.get("promotion", {}).get(
            "stable_threshold", PatternRegistry.STABLE_THRESHOLD
        )
    else:
        min_runs = getattr(policy, "min_runs", PatternRegistry.MIN_RUNS)
        working_threshold = getattr(
            policy, "working_threshold", PatternRegistry.WORKING_THRESHOLD
        )
        stable_threshold = getattr(
            policy, "stable_threshold", PatternRegistry.STABLE_THRESHOLD
        )
    is_stable = check_stability(entry["scores"])

    entry["avg_score"] = round(avg_score, 4)
    entry["confidence"] = compute_confidence(avg_score, entry["runs"], variance, min_runs)
    entry["status"] = assign_status(avg_score, is_stable, working_threshold, stable_threshold)
    entry["is_stable"] = is_stable
    entry["last_updated"] = _utc_now_iso()

    del metadata
    registry[pattern_name] = entry
    save_registry(registry, path)
    return entry


def append_run(run: dict[str, Any], path: Path = RUNS_PATH) -> None:
    runs = load_json(path, [])
    runs.append(run)
    save_json(path, runs)
