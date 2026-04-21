"""
FastAPI dashboard server.

Read-only surface over the pattern registry. Registry keys are qualified as
`<use_case>.<pattern_name>` (e.g. `concept2.erg_load_threshold`). The server
filters by splitting the key on the first `.`.

`registry.json` is derived from `memory/runs.json` and rebuilt on every
`PatternRegistry.load()`. The SSE tail watches `registry.json` mtime and
synthesizes per-run `result` events from the growth of each pattern's `scores`
list between snapshots.

NOTE: Windows file locking — on POSIX we can read registry.json while
`core.registry` rewrites it; on Windows the read may briefly fail while
the writer holds the handle. The tail swallows transient JSONDecodeError
and retries on the next poll.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

import yaml
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .ui import UI_HTML


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = REPO_ROOT / "registry.json"
DEFAULT_RUNS = REPO_ROOT / "memory" / "runs.json"
DEFAULT_POLICY = REPO_ROOT / "scoring_policy.yaml"
DEFAULT_USE_CASES_DIR = REPO_ROOT / "use_cases"

_DATASET_USE_CASE = re.compile(r"^use_cases\.([^.]+)\.")


def _use_case_from_dataset_id(dataset_id: str) -> str:
    if not dataset_id:
        return "_unknown"
    m = _DATASET_USE_CASE.match(dataset_id)
    return m.group(1) if m else "_unknown"

STATUS_TO_TIER_LABEL = {
    "bronze": "bronze",
    "silver": "silver",
    "gold": "gold",
}

USE_CASE_DISPLAY_NAMES = {
    "fraud": "Fraud Detection",
}


def _format_use_case(slug: str) -> str:
    if not slug:
        return ""
    return USE_CASE_DISPLAY_NAMES.get(slug, slug.replace("_", " ").title())


def _split_qualified(key: str) -> tuple[str, str]:
    if "." not in key:
        return ("_unknown", key)
    uc, _, name = key.partition(".")
    return (uc, name)


def list_use_cases(
    use_cases_dir: Path,
    registry_path: Path | None = None,
) -> list[dict[str, str]]:
    """List use cases from (a) subdirectories of use_cases_dir and
    (b) qualified keys actually present in registry.json."""
    slugs: set[str] = set()
    if use_cases_dir.exists():
        for child in sorted(use_cases_dir.iterdir()):
            if not child.is_dir():
                continue
            if child.name.startswith("_") or child.name.startswith("."):
                continue
            slugs.add(child.name)
    if registry_path is not None:
        raw = _load_registry_raw(registry_path)
        for key in raw.keys():
            uc, _ = _split_qualified(key)
            if uc and not uc.startswith("_"):
                slugs.add(uc)
    return [{"slug": s, "label": _format_use_case(s)} for s in sorted(slugs)]


def _normalize_use_case(value: str | None) -> str | None:
    if value is None:
        return None
    v = value.strip().lower()
    if not v or v == "all":
        return None
    return v


def detect_use_cases(registry_keys: list[str]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    unmatched: list[str] = []
    for name in registry_keys:
        slug, _ = _split_qualified(name)
        if not slug or slug.startswith("_"):
            unmatched.append(name)
            continue
        counts[slug] = counts.get(slug, 0) + 1

    dominant_slug = max(counts.items(), key=lambda kv: kv[1])[0] if counts else ""
    return {
        "current": _format_use_case(dominant_slug),
        "slug": dominant_slug,
        "all": sorted({_format_use_case(s) for s in counts.keys()}),
        "unmatched": unmatched,
    }


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
    use_case: str | None = None,
) -> dict[str, Any]:
    raw = _load_registry_raw(registry_path)
    working_t, stable_t, min_runs = _load_policy_thresholds(policy_path)

    filter_slug = _normalize_use_case(use_case)

    patterns: list[dict[str, Any]] = []
    total_runs = 0
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        entry_uc = str(entry.get("use_case", "")) or _split_qualified(name)[0]
        if filter_slug is not None and entry_uc != filter_slug:
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
        promotion_candidate = avg_score >= stable_t and runs >= min_runs and not is_stable
        pattern_name = str(entry.get("pattern_name", "")) or _split_qualified(name)[1]
        patterns.append(
            {
                "pattern": name,
                "pattern_name": pattern_name,
                "use_case": entry_uc,
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

    tier_rank = {"gold": 0, "silver": 1, "bronze": 2}
    patterns.sort(key=lambda p: (tier_rank.get(p["status"], 99), -p["confidence"]))

    use_case_info = detect_use_cases(list(raw.keys()))
    if filter_slug is not None:
        use_case_info = {
            **use_case_info,
            "selected": filter_slug,
            "selected_label": _format_use_case(filter_slug),
        }
    else:
        use_case_info = {**use_case_info, "selected": "", "selected_label": ""}

    return {
        "patterns": patterns,
        "promotion_candidates": [p["pattern"] for p in patterns if p["promotion_candidate"]],
        "thresholds": {
            "working": working_t,
            "stable": stable_t,
            "min_runs": min_runs,
        },
        "use_case": use_case_info,
        "total_experiments": total_runs,
        "last_updated": int(time.time()),
    }


def _load_runs_list(runs_path: Path) -> list[dict[str, Any]]:
    if not runs_path.exists():
        return []
    try:
        data = json.loads(runs_path.read_text())
    except (json.JSONDecodeError, ValueError, OSError):
        return []
    return data if isinstance(data, list) else []


def _ts_to_iso(unix_seconds: int) -> str:
    if unix_seconds <= 0:
        return ""
    return (
        datetime.fromtimestamp(unix_seconds, tz=timezone.utc)
        .replace(tzinfo=None)
        .isoformat(timespec="seconds")
    )


def _run_to_row(run: dict[str, Any], idx: int) -> dict[str, Any]:
    short = str(run.get("pattern", ""))
    uc = str(run.get("use_case") or _use_case_from_dataset_id(run.get("dataset_id", "")))
    qualified = f"{uc}.{short}" if short else uc
    ts = int(run.get("timestamp", 0) or 0)
    return {
        "pattern": qualified,
        "pattern_name": short,
        "use_case": uc,
        "score": float(run.get("score", 0.0) or 0.0),
        "status": str(run.get("status", "")),
        "tier": str(run.get("tier", "")),
        "dataset_id": str(run.get("dataset_id", "")),
        "commit": str(run.get("commit", "")),
        "description": str(run.get("description", "")),
        "ts": ts,
        "last_updated": _ts_to_iso(ts),
        "run_index": idx,
    }


def build_results(
    runs_path: Path,
    n: int,
    use_case: str | None = None,
) -> dict[str, Any]:
    runs = _load_runs_list(runs_path)
    filter_slug = _normalize_use_case(use_case)

    rows: list[dict[str, Any]] = []
    for idx, run in enumerate(runs):
        if not isinstance(run, dict):
            continue
        row = _run_to_row(run, idx)
        if filter_slug is not None and row["use_case"] != filter_slug:
            continue
        rows.append(row)

    rows.sort(key=lambda r: (r["ts"], r["run_index"]), reverse=True)
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


def _sse(event: str, data: Any) -> bytes:
    payload = json.dumps(data, default=str)
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


def _initial_result_events(
    runs_path: Path,
    limit: int,
    use_case: str | None = None,
) -> list[dict[str, Any]]:
    runs = _load_runs_list(runs_path)
    filter_slug = _normalize_use_case(use_case)
    rows: list[dict[str, Any]] = []
    for idx, run in enumerate(runs):
        if not isinstance(run, dict):
            continue
        row = _run_to_row(run, idx)
        if filter_slug is not None and row["use_case"] != filter_slug:
            continue
        rows.append(row)
    rows.sort(key=lambda r: (r["ts"], r["run_index"]), reverse=True)
    return rows[:limit]


async def _stream_events(
    registry_path: Path,
    policy_path: Path,
    heartbeat_interval_s: float,
    use_cases_dir: Path = DEFAULT_USE_CASES_DIR,
    poll_interval_s: float = 0.5,
    use_case: str | None = None,
    runs_path: Path = DEFAULT_RUNS,
) -> AsyncIterator[bytes]:
    snapshot = build_registry_snapshot(registry_path, policy_path, use_cases_dir, use_case)
    yield _sse("registry", snapshot)

    for ev in _initial_result_events(runs_path, limit=20, use_case=use_case):
        yield _sse("result", ev)

    filter_slug = _normalize_use_case(use_case)

    reg_mtime = registry_path.stat().st_mtime if registry_path.exists() else 0.0
    runs_mtime = runs_path.stat().st_mtime if runs_path.exists() else 0.0
    last_runs_len = len(_load_runs_list(runs_path))
    last_heartbeat = time.monotonic()

    while True:
        await asyncio.sleep(poll_interval_s)

        if (time.monotonic() - last_heartbeat) >= heartbeat_interval_s:
            yield _sse("heartbeat", {"ts": int(time.time())})
            last_heartbeat = time.monotonic()

        if runs_path.exists():
            try:
                m = runs_path.stat().st_mtime
            except OSError:
                m = runs_mtime
            if m != runs_mtime:
                runs_mtime = m
                runs = _load_runs_list(runs_path)
                if len(runs) > last_runs_len:
                    for idx in range(last_runs_len, len(runs)):
                        run = runs[idx]
                        if not isinstance(run, dict):
                            continue
                        row = _run_to_row(run, idx)
                        if filter_slug is not None and row["use_case"] != filter_slug:
                            continue
                        yield _sse("result", row)
                last_runs_len = len(runs)

        if registry_path.exists():
            try:
                m = registry_path.stat().st_mtime
            except OSError:
                m = reg_mtime
            if m != reg_mtime:
                reg_mtime = m
                new_snapshot = build_registry_snapshot(
                    registry_path, policy_path, use_cases_dir, use_case,
                )
                yield _sse("registry", new_snapshot)


def create_app(
    registry_path: Path = DEFAULT_REGISTRY,
    policy_path: Path = DEFAULT_POLICY,
    heartbeat_interval_s: float = 15.0,
    use_cases_dir: Path = DEFAULT_USE_CASES_DIR,
    runs_path: Path = DEFAULT_RUNS,
) -> FastAPI:
    app = FastAPI(title="air-lab-os dashboard", docs_url=None, redoc_url=None)
    app.state.registry_path = Path(registry_path)
    app.state.runs_path = Path(runs_path)
    app.state.policy_path = Path(policy_path)
    app.state.use_cases_dir = Path(use_cases_dir)
    app.state.heartbeat_interval_s = float(heartbeat_interval_s)

    @app.get("/", response_class=HTMLResponse)
    def root() -> HTMLResponse:
        return HTMLResponse(UI_HTML)

    @app.get("/api/registry")
    def api_registry(use_case: str | None = Query(None)) -> JSONResponse:
        snap = build_registry_snapshot(
            app.state.registry_path,
            app.state.policy_path,
            app.state.use_cases_dir,
            use_case=use_case,
        )
        return JSONResponse(snap)

    @app.get("/api/policy")
    def api_policy() -> JSONResponse:
        return JSONResponse(load_policy_dict(app.state.policy_path))

    @app.get("/api/results")
    def api_results(
        n: int = Query(50, ge=1, le=500),
        use_case: str | None = Query(None),
    ) -> JSONResponse:
        return JSONResponse(
            build_results(app.state.runs_path, n, use_case=use_case)
        )

    @app.get("/api/use_cases")
    def api_use_cases() -> JSONResponse:
        return JSONResponse({
            "use_cases": list_use_cases(app.state.use_cases_dir, app.state.registry_path),
        })

    @app.get("/api/stream")
    def api_stream(use_case: str | None = Query(None)) -> StreamingResponse:
        generator = _stream_events(
            app.state.registry_path,
            app.state.policy_path,
            app.state.heartbeat_interval_s,
            app.state.use_cases_dir,
            use_case=use_case,
            runs_path=app.state.runs_path,
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
    parser.add_argument("--runs", default=str(DEFAULT_RUNS))
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    args = parser.parse_args()

    app.state.registry_path = Path(args.registry)
    app.state.runs_path = Path(args.runs)
    app.state.policy_path = Path(args.policy)

    uvicorn.run(app, host=args.host, port=args.port)
