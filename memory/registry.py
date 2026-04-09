"""
PatternRegistry — live system state rebuilt from memory/runs.json.

Two concepts are deliberately separated:

  status — outcome of the most recent run: "keep" | "discard" | "crash"
  tier   — promotion level of the pattern:  "scratch" | "working" | "stable"

Status changes every run. Tier only changes on promotion.
Never write registry.json directly — always use PatternRegistry.save().
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


RUNS_PATH     = Path(__file__).parent / "runs.json"
REGISTRY_PATH = Path(__file__).parent.parent / "registry.json"


@dataclass
class RegistryEntry:
    # Identity
    pattern:      str              # unique pattern name
    domain:       str              # use-case domain (set by plugin)
    version:      str              # pattern version string
    pattern_path: str              # relative path to pattern file

    # Tier — promotion level, matches directory name
    tier:         str              # "scratch" | "working" | "stable"

    # Run outcome of the last experiment
    last_status:  str              # "keep" | "discard" | "crash"

    # Performance tracking
    confidence:   float            # best composite score across all keep runs
    runs:         int              # total experiment attempts
    last_commit:  str
    last_updated: int              # unix timestamp
    last_dataset: str              # dataset_id of the best scoring run

    # Promotion
    promotion_candidate: bool      # True if stable_threshold met and runs >= min_runs

    # Best run metrics — domain-agnostic dict
    best_metrics: dict[str, Any] = field(default_factory=dict)

    description: str = ""


class PatternRegistry:
    """
    Rebuilt from runs.json on every startup.
    registry.json is derived — never authoritative.

    Usage:
        registry = PatternRegistry.load()
        entry    = registry.get("my_pattern")
        registry.save()
    """

    # Thresholds — loaded from scoring policy on PatternRegistry.load()
    WORKING_THRESHOLD: float = 0.65
    STABLE_THRESHOLD:  float = 0.78
    MIN_RUNS:          int   = 3

    def __init__(self, entries: dict[str, RegistryEntry]):
        self._entries = entries

    @classmethod
    def load(
        cls,
        runs_path:     Path = RUNS_PATH,
        registry_path: Path = REGISTRY_PATH,
    ) -> "PatternRegistry":
        """
        Rebuild registry from runs.json.
        Stable tier statuses are preserved from registry.json across rebuilds.
        """
        try:
            from runtime.config import load_policy
            policy = load_policy()
            cls.WORKING_THRESHOLD = policy.working_threshold
            cls.STABLE_THRESHOLD  = policy.stable_threshold
            cls.MIN_RUNS          = policy.min_runs
        except Exception:
            pass

        stable_patterns: set[str] = set()
        if registry_path.exists():
            try:
                data = json.loads(registry_path.read_text())
                stable_patterns = {
                    k for k, v in data.items()
                    if isinstance(v, dict) and v.get("tier") == "stable"
                }
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
            score   = float(run.get("score", 0.0))
            status  = run.get("status", "discard")
            ts      = int(run.get("timestamp", 0))
            domain  = run.get("domain", "")
            dataset = run.get("dataset_id", "")

            if not pattern:
                continue

            if pattern not in entries:
                entries[pattern] = RegistryEntry(
                    pattern      = pattern,
                    domain       = domain,
                    version      = run.get("version", "0.1"),
                    pattern_path = run.get("pattern_path", ""),
                    tier         = "scratch",
                    last_status  = status,
                    confidence   = 0.0,
                    runs         = 0,
                    last_commit  = run.get("commit", ""),
                    last_updated = ts,
                    last_dataset = dataset,
                    promotion_candidate = False,
                    description  = run.get("description", ""),
                )

            e = entries[pattern]
            e.runs        += 1
            e.last_status  = status
            e.last_updated = max(e.last_updated, ts)
            e.last_commit  = run.get("commit", e.last_commit)
            e.last_dataset = dataset or e.last_dataset
            if domain:
                e.domain = domain

            if status == "keep" and score > e.confidence:
                e.confidence  = score
                e.best_metrics = run.get("metrics", {})
                e.description  = run.get("description", e.description)

        for pattern, e in entries.items():
            if pattern in stable_patterns:
                e.tier = "stable"
            elif e.confidence >= cls.WORKING_THRESHOLD:
                e.tier = "working"
            else:
                e.tier = "scratch"

            e.promotion_candidate = (
                e.confidence >= cls.STABLE_THRESHOLD
                and e.runs >= cls.MIN_RUNS
                and e.tier != "stable"
            )

        return cls(entries)

    def get(self, pattern: str) -> Optional[RegistryEntry]:
        return self._entries.get(pattern)

    def all(self) -> list[RegistryEntry]:
        return sorted(
            self._entries.values(),
            key=lambda e: e.confidence,
            reverse=True,
        )

    def by_tier(self, tier: str) -> list[RegistryEntry]:
        return [e for e in self._entries.values() if e.tier == tier]

    def best_score(self, pattern: str) -> Optional[float]:
        e = self._entries.get(pattern)
        return e.confidence if e else None

    def promotion_candidates(self) -> list[RegistryEntry]:
        return [e for e in self._entries.values() if e.promotion_candidate]

    def save(self, path: Path = REGISTRY_PATH) -> None:
        """Write registry.json. Derived — safe to overwrite."""
        data = {p: asdict(e) for p, e in self._entries.items()}
        path.write_text(json.dumps(data, indent=2))


def append_run(run: dict, path: Path = RUNS_PATH) -> None:
    """
    Append one run record to runs.json.
    Creates the file as an empty list if it does not exist.
    """
    if path.exists():
        try:
            runs = json.loads(path.read_text())
        except (json.JSONDecodeError, ValueError):
            runs = []
    else:
        runs = []

    runs.append(run)
    path.write_text(json.dumps(runs, indent=2))
