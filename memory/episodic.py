"""
runtime/memory/episodic.py — FR-006
-----------------------------------
Per-run episodic memory persisted as JSONL.

One JSON object per line, append-only. The "Episodic Memory → past runs"
layer from self-evolving-ai-system-pattern.md.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from runtime.config import MemoryConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Episode record
# ---------------------------------------------------------------------------

@dataclass
class EpisodeRecord:
    run_id:        str
    iteration:     int
    timestamp:     str
    mode:          str
    user_input:    str
    result:        dict
    evaluation:    dict
    reflection:    dict
    score:         float
    patterns_used: list[str] = field(default_factory=list)
    promoted:      list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "EpisodeRecord":
        return cls(
            run_id=data.get("run_id", ""),
            iteration=int(data.get("iteration", 0)),
            timestamp=data.get("timestamp", ""),
            mode=data.get("mode", ""),
            user_input=data.get("user_input", ""),
            result=data.get("result", {}),
            evaluation=data.get("evaluation", {}),
            reflection=data.get("reflection", {}),
            score=float(data.get("score", 0.0)),
            patterns_used=list(data.get("patterns_used", [])),
            promoted=list(data.get("promoted", [])),
        )


# ---------------------------------------------------------------------------
# EpisodicMemory
# ---------------------------------------------------------------------------

class EpisodicMemory:
    """JSONL-backed episodic memory store. Thread-safe via in-process lock."""

    def __init__(self, config: MemoryConfig) -> None:
        self._config = config
        self._path = Path(config.episodic_path)
        self._enabled = config.episodic_enabled
        self._lock = threading.Lock()
        if self._enabled:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------
    # Write
    # ----------------------------------------------------------------------

    def record(self, episode: EpisodeRecord) -> None:
        """Append one episode to the JSONL store."""
        if not self._enabled:
            return
        with self._lock:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(episode)) + "\n")

    # ----------------------------------------------------------------------
    # Read
    # ----------------------------------------------------------------------

    def _iter_records(self) -> list[EpisodeRecord]:
        if not self._enabled or not self._path.exists():
            return []
        records: list[EpisodeRecord] = []
        with self._path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(EpisodeRecord.from_dict(json.loads(line)))
                except json.JSONDecodeError as exc:
                    log.warning("episodic: bad line skipped (%s)", exc)
        return records

    def _cap(self, records: list[EpisodeRecord]) -> list[EpisodeRecord]:
        return records[: self._config.episodic_max_entries]

    def load_run(self, run_id: str) -> list[EpisodeRecord]:
        return self._cap([r for r in self._iter_records() if r.run_id == run_id])

    def load_recent(self, n: int = 10) -> list[EpisodeRecord]:
        if not self._enabled:
            return []
        records = self._iter_records()
        return self._cap(records[-n:])

    def load_by_score(self, min_score: float) -> list[EpisodeRecord]:
        return self._cap([r for r in self._iter_records() if r.score >= min_score])

    # ----------------------------------------------------------------------
    # Rotation
    # ----------------------------------------------------------------------

    def rotate(self) -> int:
        """Drop episodes older than retention_days. Returns count deleted."""
        if not self._enabled or not self._path.exists():
            return 0
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._config.episodic_retention_days)
        kept: list[EpisodeRecord] = []
        deleted = 0
        for record in self._iter_records():
            if self._is_recent(record.timestamp, cutoff):
                kept.append(record)
            else:
                deleted += 1
        if deleted:
            with self._lock:
                with self._path.open("w", encoding="utf-8") as fh:
                    for r in kept:
                        fh.write(json.dumps(asdict(r)) + "\n")
            log.info("episodic: rotated %d expired entries", deleted)
        return deleted

    @staticmethod
    def _is_recent(ts: str, cutoff: datetime) -> bool:
        try:
            parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed >= cutoff
        except (ValueError, TypeError):
            return True  # keep on parse failure
