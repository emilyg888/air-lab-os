"""
runtime/memory/run_store.py — FR-04
-----------------------------------
Persistent store of completed run summaries.

One record per completed run (not per iteration). Used by the engine to
seed the next run with what worked in previous runs — the "Rolling" half
of TheRollingPipelines.

Separate from episodic.jsonl (per-iteration).
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class RunSummary:
    """Compact summary of a completed run."""
    run_id:          str
    timestamp:       str
    mode:            str
    user_input:      str
    final_score:     float
    iterations:      int
    patterns_used:   list[str] = field(default_factory=list)
    promoted:        list[str] = field(default_factory=list)
    graduated:       list[str] = field(default_factory=list)
    best_tasks:      list[str] = field(default_factory=list)
    best_approaches: list[str] = field(default_factory=list)
    strategy:        str = ""


class RunStore:
    """JSONL store of run summaries — one record per completed run."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------
    # Write
    # ----------------------------------------------------------------------

    def save(self, summary: RunSummary) -> None:
        with self._lock:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(summary)) + "\n")
        log.info(
            "run_store: saved run %s (score=%.3f)",
            summary.run_id, summary.final_score,
        )

    # ----------------------------------------------------------------------
    # Read
    # ----------------------------------------------------------------------

    def _iter_records(self) -> list[RunSummary]:
        if not self._path.exists():
            return []
        records: list[RunSummary] = []
        with self._path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    records.append(RunSummary(**data))
                except (json.JSONDecodeError, TypeError) as exc:
                    log.warning("run_store: skipping bad line (%s)", exc)
                    continue
        return records

    def load_recent_successes(
        self,
        n: int = 3,
        min_score: float = 0.80,
    ) -> list[RunSummary]:
        """Return the n most recent runs with score >= min_score."""
        records = self._iter_records()
        qualifying = [r for r in records if r.final_score >= min_score]
        return qualifying[-n:]

    def load_all(self) -> list[RunSummary]:
        return self._iter_records()
