"""
Pattern registry — live system state rebuilt from results.tsv.

PatternRegistry is rebuilt from the TSV on every startup.
Never write registry.json directly — always go through PatternRegistry.save().

Status values:
  bronze  — score >= 0 but below silver_threshold
  silver  — score >= silver_threshold
  gold    — manually promoted (promotion_candidate surfaced to human)
  running — currently being evaluated
  discard — ran but did not improve
  crash   — experiment crashed
"""

from __future__ import annotations
import json
import csv
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class RegistryEntry:
    pattern: str
    status: str                     # bronze | silver | gold | running | discard | crash
    confidence: float               # best score achieved
    runs: int                       # total experiment count
    last_commit: str
    last_updated: int               # unix timestamp
    promotion_candidate: bool       # True if score >= gold_threshold and runs >= min_runs
    best_precision: float = 0.0
    best_recall: float = 0.0
    best_f1: float = 0.0
    description: str = ""


class PatternRegistry:
    """
    Manages pattern state. Rebuilt from results.tsv on load.

    Usage:
        registry = PatternRegistry.load(registry_path, results_path)
        entry = registry.get("velocity")
        registry.save(registry_path)
    """

    # These thresholds mirror scoring_policy.yaml.
    # They are read from the policy file on load — do not hardcode here.
    SILVER_THRESHOLD: float = 0.65
    GOLD_THRESHOLD: float = 0.78
    MIN_RUNS: int = 3

    def __init__(self, entries: dict[str, RegistryEntry]):
        self._entries = entries

    @classmethod
    def load(
        cls,
        registry_path: Path,
        results_path: Path,
    ) -> "PatternRegistry":
        """
        Rebuild registry from results.tsv.
        Ignores registry.json — TSV is authoritative.
        """
        # Load promotion thresholds from scoring policy
        try:
            from src.engine.evaluate import load_policy
            policy = load_policy()
            cls.SILVER_THRESHOLD = policy.silver_threshold
            cls.GOLD_THRESHOLD = policy.gold_threshold
            cls.MIN_RUNS = policy.min_runs
        except Exception:
            pass  # use class defaults if policy unavailable

        # Load existing gold statuses from registry.json (only gold is preserved)
        gold_patterns: set[str] = set()
        if registry_path.exists():
            try:
                data = json.loads(registry_path.read_text())
                gold_patterns = {
                    k for k, v in data.items()
                    if isinstance(v, dict) and v.get("status") == "gold"
                }
            except Exception:
                pass

        entries: dict[str, RegistryEntry] = {}

        if not results_path.exists():
            return cls(entries)

        with open(results_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                pattern = row["pattern"]
                score = float(row.get("score", 0))
                status = row.get("status", "discard")
                ts = int(row.get("timestamp", 0))

                if pattern not in entries:
                    entries[pattern] = RegistryEntry(
                        pattern=pattern,
                        status="bronze",
                        confidence=0.0,
                        runs=0,
                        last_commit=row.get("commit", ""),
                        last_updated=ts,
                        promotion_candidate=False,
                        description=row.get("description", ""),
                    )

                e = entries[pattern]
                e.runs += 1
                e.last_updated = max(e.last_updated, ts)
                e.last_commit = row.get("commit", e.last_commit)

                if status in ("keep",) and score > e.confidence:
                    e.confidence = score
                    e.best_precision = float(row.get("precision", 0))
                    e.best_recall = float(row.get("recall", 0))
                    e.best_f1 = float(row.get("f1", 0))

        # Compute statuses
        for pattern, e in entries.items():
            if pattern in gold_patterns:
                e.status = "gold"
            elif e.confidence >= cls.SILVER_THRESHOLD:
                e.status = "silver"
            else:
                e.status = "bronze"

            e.promotion_candidate = (
                e.confidence >= cls.GOLD_THRESHOLD
                and e.runs >= cls.MIN_RUNS
                and e.status != "gold"
            )

        return cls(entries)

    def get(self, pattern: str) -> Optional[RegistryEntry]:
        return self._entries.get(pattern)

    def all(self) -> list[RegistryEntry]:
        return sorted(self._entries.values(), key=lambda e: e.confidence, reverse=True)

    def best_score(self, pattern: str) -> Optional[float]:
        e = self._entries.get(pattern)
        return e.confidence if e else None

    def promotion_candidates(self) -> list[RegistryEntry]:
        return [e for e in self._entries.values() if e.promotion_candidate]

    def save(self, path: Path) -> None:
        """Write registry.json. Derived from TSV — safe to overwrite."""
        data = {pattern: asdict(entry) for pattern, entry in self._entries.items()}
        path.write_text(json.dumps(data, indent=2))
