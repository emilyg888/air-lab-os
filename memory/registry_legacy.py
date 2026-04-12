"""Derived pattern registry rebuilt from memory/runs.json."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


RUNS_PATH = Path(__file__).parent / "runs.json"
REGISTRY_PATH = Path(__file__).parent.parent / "registry.json"

STATUS_TO_TIER = {
    "bronze": "scratch",
    "silver": "working",
    "gold": "stable",
}
TIER_TO_STATUS = {value: key for key, value in STATUS_TO_TIER.items()}
STABILITY_VARIANCE_THRESHOLD = 0.0025


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

    # Runtime metadata kept in memory for compatibility with existing callers.
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
    """Rebuilt from runs.json on every startup."""

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
        del registry_path

        try:
            from runtime.config import load_policy

            policy = load_policy()
            cls.WORKING_THRESHOLD = policy.working_threshold
            cls.STABLE_THRESHOLD = policy.stable_threshold
            cls.MIN_RUNS = policy.min_runs
        except Exception:
            pass

        entries: dict[str, RegistryEntry] = {}

        if not runs_path.exists():
            return cls(entries)

        try:
            runs = json.loads(runs_path.read_text())
        except (json.JSONDecodeError, ValueError):
            runs = []

        for run in runs:
            pattern = run.get("pattern", "")
            if not pattern:
                continue

            score = float(run.get("score", 0.0))
            run_status = run.get("status", "discard")
            timestamp = int(run.get("timestamp", 0))

            if pattern not in entries:
                entries[pattern] = RegistryEntry(
                    pattern=pattern,
                    domain=run.get("domain", ""),
                    version=run.get("version", "0.1"),
                    pattern_path=run.get("pattern_path", ""),
                )

            entry = entries[pattern]
            entry.runs += 1
            entry.scores.append(score)
            entry.last_score = score
            entry.last_run_status = run_status
            entry.last_commit = run.get("commit", entry.last_commit)
            entry.last_dataset = run.get("dataset_id", entry.last_dataset)
            entry.last_updated = _iso_timestamp(timestamp)

            if run.get("domain"):
                entry.domain = run["domain"]
            if run.get("version"):
                entry.version = run["version"]
            if run.get("pattern_path"):
                entry.pattern_path = run["pattern_path"]

            if score >= entry.max_score:
                entry.best_metrics = run.get("metrics", {})
                entry.description = run.get("description", entry.description)

        for entry in entries.values():
            entry.avg_score = _mean(entry.scores)
            variance = _variance(entry.scores)
            sample_factor = min(1.0, entry.runs / max(cls.MIN_RUNS, 1))
            stability_factor = max(0.0, 1.0 - (variance * 20))
            entry.confidence = round(entry.avg_score * sample_factor * stability_factor, 4)
            entry.is_stable = (
                entry.runs >= cls.MIN_RUNS
                and variance <= STABILITY_VARIANCE_THRESHOLD
            )
            entry.status = _assign_status(entry.avg_score, entry.is_stable, cls)

        return cls(entries)

    def get(self, pattern: str) -> Optional[RegistryEntry]:
        return self._entries.get(pattern)

    def all(self) -> list[RegistryEntry]:
        return sorted(
            self._entries.values(),
            key=lambda entry: entry.confidence,
            reverse=True,
        )

    def by_tier(self, tier: str) -> list[RegistryEntry]:
        target_status = TIER_TO_STATUS.get(tier, tier)
        return [entry for entry in self._entries.values() if entry.status == target_status]

    def best_score(self, pattern: str) -> Optional[float]:
        entry = self._entries.get(pattern)
        return entry.max_score if entry else None

    def promotion_candidates(self) -> list[RegistryEntry]:
        return [entry for entry in self._entries.values() if entry.promotion_candidate]

    def save(self, path: Path = REGISTRY_PATH) -> None:
        data = {pattern: entry.to_dict() for pattern, entry in self._entries.items()}
        path.write_text(json.dumps(data, indent=2))


def _assign_status(avg_score: float, is_stable: bool, registry_cls: type[PatternRegistry]) -> str:
    if avg_score >= registry_cls.STABLE_THRESHOLD and is_stable:
        return "gold"
    if avg_score >= registry_cls.WORKING_THRESHOLD:
        return "silver"
    return "bronze"


def append_run(run: dict, path: Path = RUNS_PATH) -> None:
    """Append one run record to runs.json."""
    if path.exists():
        try:
            runs = json.loads(path.read_text())
        except (json.JSONDecodeError, ValueError):
            runs = []
    else:
        runs = []

    runs.append(run)
    path.write_text(json.dumps(runs, indent=2))
