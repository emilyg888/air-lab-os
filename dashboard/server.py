"""
FastAPI dashboard server.

Read-only surface over the pattern registry. Adapted for air-lab-os:
- The source of truth is `registry.json` (per-pattern aggregated scores list),
  incrementally written by `core.registry.update_registry()`.
- There is no append-only `results.tsv` in this repo — the SSE tail watches
  `registry.json` mtime and synthesizes per-run `result` events from the
  growth of each pattern's `scores` list between snapshots.

NOTE: Windows file locking — on POSIX we can read registry.json while
`core.registry` rewrites it; on Windows the read may briefly fail while
the writer holds the handle. The tail swallows transient JSONDecodeError
and retries on the next poll.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, AsyncIterator

import yaml
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .ui import UI_HTML


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = REPO_ROOT / "registry.json"
DEFAULT_POLICY = REPO_ROOT / "scoring_policy.yaml"
DEFAULT_USE_CASES_DIR = REPO_ROOT / "use_cases"

# Map core.registry status → tier label used by the UI badges.
STATUS_TO_TIER_LABEL = {
    "bronze": "bronze",
    "silver": "silver",
    "gold": "gold",
}

# Friendly names for well-known use-case slugs. Falls back to Title Case.
USE_CASE_DISPLAY_NAMES = {
    "fraud": "Fraud Detection",
}


def _scan_use_case_patterns(use_cases_dir: Path) -> dict[str, str]:
    """
    Walk use_cases/*/patterns/*.py and return a map of
    pattern_name (file stem) → use_case slug (directory name).
    """
    mapping: dict[str, str] = {}
    if not use_cases_dir.exists():
        return mapping
    for pattern_file in use_cases_dir.glob("*/patterns/*.py"):
        if pattern_file.name.startswith("_"):
            continue
        use_case = pattern_file.parent.parent.name
        mapping[pattern_file.stem] = use_case
    return mapping


def _format_use_case(slug: str) -> str:
    if not slug:
        return ""
    return USE_CASE_DISPLAY_NAMES.get(slug, slug.replace("_", " ").title())


def detect_use_cases(
    registry_keys: list[str],
    use_cases_dir: Path = DEFAULT_USE_CASES_DIR,
) -> dict[str, Any]:
    """
    Figure out which use case(s) the registered patterns belong to by
    matching pattern names against use_cases/*/patterns/*.py files.

    Returns:
        {
          "current": "Fraud Detection",   # dominant use case (most patterns)
          "slug": "fraud",
          "all": ["Fraud Detection"],     # every matched use case
          "unmatched": ["foo_pattern"],   # registry entries with no match
        }
    """
    pattern_to_use_case = _scan_use_case_patterns(use_cases_dir)
    counts: dict[str, int] = {}
    unmatched: list[str] = []
    for name in registry_keys:
        slug = pattern_to_use_case.get(name)
        if slug is None:
            unmatched.append(name)
            continue
        counts[slug] = counts.get(slug, 0) + 1

    if counts:
        dominant_slug = max(counts.items(), key=lambda kv: kv[1])[0]
    else:
        dominant_slug = ""

    return {
        "current": _format_use_case(dominant_slug),
        "slug": dominant_slug,
        "all": sorted({_format_use_case(s) for s in counts.keys()}),
        "unmatched": unmatched,
    }


# ---------------------------------------------------------------------------
# Registry / policy snapshot helpers
# ---------------------------------------------------------------------------


def _load_registry_raw(registry_path: Path) -> dict[str, Any]:
    if not registry_path.exists():
        return {}
    try:
        data = json.loads(registry_path.read_text())
    except (json.JSONDecodeError, ValueError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _load_policy_thresholds(policy_path: Path) -> tuple[float, float, int]:
    try:
        raw = yaml.safe_load(policy_path.read_text())
        promo = raw.get("promotion", {})
        return (
            float(promo.get("working_threshold", 0.65)),
            float(promo.get("stable_threshold", 0.78)),
            int(promo.get("min_runs", 3)),
        )
    except Exception:
        return 0.65, 0.78, 3


def _normalize_last_updated(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def build_registry_snapshot(
    registry_path: Path,
    policy_path: Path,
    use_cases_dir: Path = DEFAULT_USE_CASES_DIR,
) -> dict[str, Any]:
    """
    Build the /api/registry response payload directly from registry.json.
    Sorted descending by confidence.
    """
    raw = _load_registry_raw(registry_path)
    working_t, stable_t, min_runs = _load_policy_thresholds(policy_path)

    patterns: list[dict[str, Any]] = []
    total_runs = 0
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        runs = int(entry.get("runs", 0))
        total_runs += runs
        scores = entry.get("scores", [])
        if not isinstance(scores, list):
            scores = []
        status = str(entry.get("status", "bronze"))
        avg_score = float(entry.get("avg_score", 0.0))
        confidence = float(entry.get("confidence", 0.0))
        is_stable = bool(entry.get("is_stable", False))
        promotion_candidate = (
            avg_score >= stable_t
            and runs >= min_runs
            and not is_stable
        )
        patterns.append(
            {
                "pattern": name,
                "status": status,
                "tier": STATUS_TO_TIER_LABEL.get(status, status),
                "confidence": round(confidence, 4),
                "avg_score": round(avg_score, 4),
                "last_score": float(entry.get("last_score", 0.0)),
                "runs": runs,
                "scores": [float(s) for s in scores],
                "is_stable": is_stable,
                "promotion_candidate": promotion_candidate,
                "last_updated": _normalize_last_updated(entry.get("last_updated")),
            }
        )

    # Primary sort: tier (gold → silver → bronze → other).
    # Secondary sort: confidence descending within the tier.
    tier_rank = {"gold": 0, "silver": 1, "bronze": 2}
    patterns.sort(
        key=lambda p: (tier_rank.get(p["status"], 99), -p["confidence"])
    )

    use_case_info = detect_use_cases(list(raw.keys()), use_cases_dir)

    return {
        "patterns": patterns,
        "promotion_candidates": [
            p["pattern"] for p in patterns if p["promotion_candidate"]
        ],
        "thresholds": {
            "working": working_t,
            "stable": stable_t,
            "min_runs": min_runs,
        },
        "use_case": use_case_info,
        "total_experiments": total_runs,
        "last_updated": int(time.time()),
    }


def build_results(registry_path: Path, n: int) -> dict[str, Any]:
    """
    Synthesize an experiment row list from per-pattern score arrays.

    Each element of a pattern's `scores` list becomes one row. Rows are
    ordered by metric score descending so the strongest experiment
    outcomes stay at the top of the log.
    """
    raw = _load_registry_raw(registry_path)
    rows: list[dict[str, Any]] = []

    for name, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        scores = entry.get("scores") or []
        if not isinstance(scores, list):
            continue
        last_updated = _normalize_last_updated(entry.get("last_updated"))
        status = str(entry.get("status", "bronze"))
        for idx, score in enumerate(scores):
            rows.append(
                {
                    "pattern": name,
                    "run_index": idx,
                    "score": float(score),
                    "status": status,
                    "last_updated": last_updated,
                    "is_last": idx == len(scores) - 1,
                }
            )

    rows.sort(
        key=lambda r: (r["score"], r["last_updated"], r["is_last"], r["run_index"]),
        reverse=True,
    )

    total = len(rows)
    limited = rows[:n]
    return {"rows": limited, "total": total, "returned": len(limited)}


def load_policy_dict(policy_path: Path) -> dict[str, Any]:
    if not policy_path.exists():
        return {}
    try:
        parsed = yaml.safe_load(policy_path.read_text())
    except yaml.YAMLError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _sse(event: str, data: Any) -> bytes:
    payload = json.dumps(data, default=str)
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


def _scores_by_pattern(snapshot: dict[str, Any]) -> dict[str, list[float]]:
    return {p["pattern"]: list(p.get("scores", [])) for p in snapshot["patterns"]}


def _diff_result_events(
    prev: dict[str, list[float]], curr: dict[str, list[float]], curr_snapshot: dict[str, Any]
) -> list[dict[str, Any]]:
    """Yield one event dict per newly-appended score since the last snapshot."""
    events: list[dict[str, Any]] = []
    status_by_pattern = {p["pattern"]: p for p in curr_snapshot["patterns"]}
    for name, scores in curr.items():
        prev_scores = prev.get(name, [])
        if len(scores) <= len(prev_scores):
            continue
        info = status_by_pattern.get(name, {})
        for idx in range(len(prev_scores), len(scores)):
            events.append(
                {
                    "pattern": name,
                    "run_index": idx,
                    "score": float(scores[idx]),
                    "status": info.get("status", "bronze"),
                    "last_updated": info.get("last_updated", ""),
                    "ts": int(time.time()),
                }
            )
    return events


def _initial_result_events(snapshot: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    """Top N synthesized result events across the history by score descending."""
    rows: list[dict[str, Any]] = []
    for p in snapshot["patterns"]:
        scores = p.get("scores") or []
        for idx, score in enumerate(scores):
            rows.append(
                {
                    "pattern": p["pattern"],
                    "run_index": idx,
                    "score": float(score),
                    "status": p.get("status", "bronze"),
                    "last_updated": p.get("last_updated", ""),
                    "ts": 0,
                    "is_last": idx == len(scores) - 1,
                }
            )
    rows.sort(
        key=lambda r: (r["score"], r["last_updated"], r["is_last"], r["run_index"]),
        reverse=True,
    )
    return rows[:limit]


async def _stream_events(
    registry_path: Path,
    policy_path: Path,
    heartbeat_interval_s: float,
    use_cases_dir: Path = DEFAULT_USE_CASES_DIR,
    poll_interval_s: float = 0.5,
) -> AsyncIterator[bytes]:
    # --- On-connect: registry snapshot + last 10 result events ---
    snapshot = build_registry_snapshot(registry_path, policy_path, use_cases_dir)
    yield _sse("registry", snapshot)

    for ev in _initial_result_events(snapshot, limit=10):
        yield _sse("result", ev)

    last_mtime = (
        registry_path.stat().st_mtime if registry_path.exists() else 0.0
    )
    last_scores = _scores_by_pattern(snapshot)
    last_heartbeat = time.monotonic()

    while True:
        await asyncio.sleep(poll_interval_s)

        # Heartbeat even if nothing changed.
        if (time.monotonic() - last_heartbeat) >= heartbeat_interval_s:
            yield _sse("heartbeat", {"ts": int(time.time())})
            last_heartbeat = time.monotonic()

        if not registry_path.exists():
            continue

        try:
            mtime = registry_path.stat().st_mtime
        except OSError:
            continue

        if mtime == last_mtime:
            continue
        last_mtime = mtime

        new_snapshot = build_registry_snapshot(registry_path, policy_path, use_cases_dir)
        new_scores = _scores_by_pattern(new_snapshot)

        for ev in _diff_result_events(last_scores, new_scores, new_snapshot):
            yield _sse("result", ev)

        yield _sse("registry", new_snapshot)
        last_scores = new_scores


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    registry_path: Path = DEFAULT_REGISTRY,
    policy_path: Path = DEFAULT_POLICY,
    heartbeat_interval_s: float = 15.0,
    use_cases_dir: Path = DEFAULT_USE_CASES_DIR,
) -> FastAPI:
    app = FastAPI(title="air-lab-os dashboard", docs_url=None, redoc_url=None)
    app.state.registry_path = Path(registry_path)
    app.state.policy_path = Path(policy_path)
    app.state.use_cases_dir = Path(use_cases_dir)
    app.state.heartbeat_interval_s = float(heartbeat_interval_s)

    @app.get("/", response_class=HTMLResponse)
    def root() -> HTMLResponse:
        return HTMLResponse(UI_HTML)

    @app.get("/api/registry")
    def api_registry() -> JSONResponse:
        snap = build_registry_snapshot(
            app.state.registry_path,
            app.state.policy_path,
            app.state.use_cases_dir,
        )
        return JSONResponse(snap)

    @app.get("/api/policy")
    def api_policy() -> JSONResponse:
        return JSONResponse(load_policy_dict(app.state.policy_path))

    @app.get("/api/results")
    def api_results(n: int = Query(50, ge=1, le=500)) -> JSONResponse:
        return JSONResponse(build_results(app.state.registry_path, n))

    @app.get("/api/stream")
    def api_stream() -> StreamingResponse:
        generator = _stream_events(
            app.state.registry_path,
            app.state.policy_path,
            app.state.heartbeat_interval_s,
            app.state.use_cases_dir,
        )
        return StreamingResponse(generator, media_type="text/event-stream")

    return app


app = create_app()


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    args = parser.parse_args()

    app.state.registry_path = Path(args.registry)
    app.state.policy_path = Path(args.policy)

    uvicorn.run(app, host=args.host, port=args.port)
