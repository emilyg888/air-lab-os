"""Signal-memory helpers for fraud feature contribution tracking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FEATURE_CONTRIBUTION_LOG_PATH = (
    Path(__file__).resolve().parent / "feature_contribution_log.json"
)


def load_feature_contribution_log(
    path: Path = FEATURE_CONTRIBUTION_LOG_PATH,
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_feature_contribution_log(
    log: dict[str, dict[str, Any]],
    path: Path = FEATURE_CONTRIBUTION_LOG_PATH,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log, indent=2, sort_keys=True) + "\n")
    return path
