"""Integration tests for the Phase 3 dashboard server."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dashboard.server import create_app


POLICY_YAML = """\
version: "1.0"
weights:
  primary_metric:  0.40
  explainability:  0.25
  latency:         0.20
  cost:            0.15
promotion:
  working_threshold: 0.65
  stable_threshold:  0.78
  min_runs:          3
latency:
  max_ms: 500
cost:
  max_per_1k: 0.10
"""


def _write_policy(tmp_path: Path) -> Path:
    p = tmp_path / "scoring_policy.yaml"
    p.write_text(POLICY_YAML)
    return p


def _write_registry(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))


def _write_runs(path: Path, runs: list[dict]) -> None:
    path.write_text(json.dumps(runs, indent=2))


def _run_rec(pattern: str, use_case: str, score: float, ts: int, status: str = "keep") -> dict:
    return {
        "pattern": pattern,
        "use_case": use_case,
        "dataset_id": f"use_cases.{use_case}.handle",
        "score": score,
        "status": status,
        "tier": "scratch",
        "commit": "testsha",
        "timestamp": ts,
        "description": f"{use_case}.{pattern} run",
    }


def _make_use_cases_tree(tmp_path: Path) -> Path:
    root = tmp_path / "use_cases"
    fraud = root / "fraud" / "patterns"
    fraud.mkdir(parents=True)
    for name in ("rule_velocity.py", "ml_logistic.py", "rule_spike.py"):
        (fraud / name).write_text("")
    (fraud / "__init__.py").write_text("")
    concept2 = root / "concept2" / "patterns"
    concept2.mkdir(parents=True)
    (concept2 / "stroke_cadence.py").write_text("")
    (concept2 / "__init__.py").write_text("")
    return root


@pytest.fixture
def paths(tmp_path):
    return {
        "registry": tmp_path / "registry.json",
        "runs": tmp_path / "runs.json",
        "policy": _write_policy(tmp_path),
        "use_cases": _make_use_cases_tree(tmp_path),
    }


@pytest.fixture
def client(paths):
    app = create_app(
        registry_path=paths["registry"],
        policy_path=paths["policy"],
        heartbeat_interval_s=0.3,
        use_cases_dir=paths["use_cases"],
        runs_path=paths["runs"],
    )
    return TestClient(app)


def test_root_returns_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "AIR LAB OS" in r.text
    assert 'id="detail-modal"' in r.text
    assert 'id="log-title"' in r.text


def test_registry_empty(client, paths):
    assert not paths["registry"].exists()
    r = client.get("/api/registry")
    assert r.status_code == 200
    body = r.json()
    assert body["patterns"] == []
    assert body["promotion_candidates"] == []
    assert body["total_experiments"] == 0


def test_registry_reflects_file(client, paths):
    _write_registry(
        paths["registry"],
        {
            "velocity": {
                "runs": 4,
                "scores": [0.70, 0.71, 0.72, 0.73],
                "avg_score": 0.715,
                "last_score": 0.73,
                "confidence": 0.705,
                "status": "silver",
                "is_stable": False,
                "last_updated": "2026-04-09T10:00:00",
            },
            "logistic": {
                "runs": 3,
                "scores": [0.90, 0.91, 0.92],
                "avg_score": 0.91,
                "last_score": 0.92,
                "confidence": 0.905,
                "status": "gold",
                "is_stable": True,
                "last_updated": "2026-04-09T11:00:00",
            },
        },
    )
    r = client.get("/api/registry")
    body = r.json()
    assert len(body["patterns"]) == 2
    assert body["patterns"][0]["pattern"] == "logistic"
    assert body["patterns"][0]["status"] == "gold"
    assert body["patterns"][1]["pattern"] == "velocity"
    assert body["total_experiments"] == 7


def test_registry_sorted_by_tier(client, paths):
    _write_registry(
        paths["registry"],
        {
            "bronze_high": {
                "runs": 2, "scores": [0.99, 0.99], "avg_score": 0.99,
                "last_score": 0.99, "confidence": 0.99, "status": "bronze",
            },
            "silver_mid": {
                "runs": 2, "scores": [0.70, 0.70], "avg_score": 0.70,
                "last_score": 0.70, "confidence": 0.70, "status": "silver",
            },
            "gold_low": {
                "runs": 3, "scores": [0.80, 0.80, 0.80], "avg_score": 0.80,
                "last_score": 0.80, "confidence": 0.80, "status": "gold",
                "is_stable": True,
            },
        },
    )
    body = client.get("/api/registry").json()
    order = [p["pattern"] for p in body["patterns"]]
    assert order == ["gold_low", "silver_mid", "bronze_high"]


def test_results_endpoint_returns_run_history(client, paths):
    _write_runs(
        paths["runs"],
        [
            _run_rec("rule_velocity", "fraud", 0.60, ts=1700000100),
            _run_rec("rule_velocity", "fraud", 0.65, ts=1700000200),
            _run_rec("rule_velocity", "fraud", 0.70, ts=1700000300),
            _run_rec("ml_logistic", "fraud", 0.80, ts=1700000400),
            _run_rec("ml_logistic", "fraud", 0.82, ts=1700000500),
        ],
    )
    body = client.get("/api/results?n=2").json()
    assert body["total"] == 5
    assert body["returned"] == 2
    assert len(body["rows"]) == 2
    # Most-recent-first by timestamp.
    assert body["rows"][0]["pattern"] == "fraud.ml_logistic"
    assert body["rows"][0]["score"] == pytest.approx(0.82)
    assert body["rows"][0]["ts"] == 1700000500
    assert body["rows"][0]["use_case"] == "fraud"
    assert body["rows"][0]["commit"] == "testsha"
    assert body["rows"][1]["ts"] == 1700000400


def test_registry_exposes_use_case(client, paths):
    _write_registry(
        paths["registry"],
        {
            "fraud.rule_velocity": {
                "use_case": "fraud", "pattern_name": "rule_velocity",
                "runs": 2, "scores": [0.6, 0.7], "avg_score": 0.65,
                "last_score": 0.7, "confidence": 0.6, "status": "silver",
                "last_updated": "2026-04-09T10:00:00",
            },
            "fraud.ml_logistic": {
                "use_case": "fraud", "pattern_name": "ml_logistic",
                "runs": 3, "scores": [0.9, 0.91, 0.92], "avg_score": 0.91,
                "last_score": 0.92, "confidence": 0.9, "status": "gold",
                "is_stable": True, "last_updated": "2026-04-09T11:00:00",
            },
        },
    )
    body = client.get("/api/registry").json()
    assert "use_case" in body
    assert body["use_case"]["slug"] == "fraud"
    assert body["use_case"]["current"] == "Fraud Detection"
    assert body["use_case"]["unmatched"] == []


def test_use_cases_endpoint_lists_subdirectories(client):
    r = client.get("/api/use_cases")
    assert r.status_code == 200
    body = r.json()
    slugs = [u["slug"] for u in body["use_cases"]]
    assert slugs == ["concept2", "fraud"]
    labels = {u["slug"]: u["label"] for u in body["use_cases"]}
    assert labels["fraud"] == "Fraud Detection"
    assert labels["concept2"] == "Concept2"


def test_registry_filters_by_use_case(client, paths):
    _write_registry(
        paths["registry"],
        {
            "fraud.rule_velocity": {
                "use_case": "fraud", "pattern_name": "rule_velocity",
                "runs": 2, "scores": [0.6, 0.7], "avg_score": 0.65,
                "last_score": 0.7, "confidence": 0.6, "status": "silver",
            },
            "concept2.stroke_cadence": {
                "use_case": "concept2", "pattern_name": "stroke_cadence",
                "runs": 3, "scores": [0.8, 0.82, 0.84], "avg_score": 0.82,
                "last_score": 0.84, "confidence": 0.8, "status": "gold",
                "is_stable": True,
            },
        },
    )
    fraud_only = client.get("/api/registry?use_case=fraud").json()
    assert [p["pattern"] for p in fraud_only["patterns"]] == ["fraud.rule_velocity"]
    assert fraud_only["total_experiments"] == 2
    assert fraud_only["use_case"]["selected"] == "fraud"

    concept2_only = client.get("/api/registry?use_case=concept2").json()
    assert [p["pattern"] for p in concept2_only["patterns"]] == ["concept2.stroke_cadence"]

    all_view = client.get("/api/registry?use_case=all").json()
    assert {p["pattern"] for p in all_view["patterns"]} == {
        "fraud.rule_velocity", "concept2.stroke_cadence",
    }
    assert all_view["use_case"]["selected"] == ""


def test_results_filters_by_use_case(client, paths):
    _write_runs(
        paths["runs"],
        [
            _run_rec("rule_velocity", "fraud", 0.6, ts=1700000100),
            _run_rec("rule_velocity", "fraud", 0.7, ts=1700000200),
            _run_rec("stroke_cadence", "concept2", 0.8, ts=1700000300),
            _run_rec("stroke_cadence", "concept2", 0.85, ts=1700000400),
        ],
    )
    body = client.get("/api/results?n=10&use_case=concept2").json()
    assert body["total"] == 2
    assert all(r["use_case"] == "concept2" for r in body["rows"])
    assert all(r["pattern"] == "concept2.stroke_cadence" for r in body["rows"])


def test_policy_endpoint(client):
    r = client.get("/api/policy")
    assert r.status_code == 200
    body = r.json()
    assert "weights" in body
    assert body["weights"]["primary_metric"] == 0.40
    assert body["promotion"]["working_threshold"] == 0.65


def _parse_sse_block(block: bytes) -> tuple[str | None, str]:
    ev_type: str | None = None
    data_lines: list[str] = []
    for line in block.split(b"\n"):
        if line.startswith(b"event: "):
            ev_type = line[len(b"event: "):].decode("utf-8").strip()
        elif line.startswith(b"data: "):
            data_lines.append(line[len(b"data: "):].decode("utf-8"))
    return ev_type, "\n".join(data_lines)


async def _collect_sse(
    app,
    min_events: int,
    wanted_types: set[str] | None = None,
    timeout_s: float = 4.0,
):
    messages: list[dict] = []
    loop = asyncio.get_running_loop()
    finished = loop.create_future()

    async def receive():
        await asyncio.sleep(3600)
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] != "http.response.body":
            return
        body = message.get("body", b"")
        chunks = body.split(b"\n\n")
        for chunk in chunks:
            if not chunk.strip():
                continue
            ev_type, payload = _parse_sse_block(chunk + b"\n")
            if wanted_types and ev_type not in wanted_types:
                continue
            messages.append({"event": ev_type, "data": json.loads(payload)})
            if len(messages) >= min_events and not finished.done():
                finished.set_result(None)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/api/stream",
        "raw_path": b"/api/stream",
        "query_string": b"",
        "headers": [],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }

    task = asyncio.create_task(app(scope, receive, send))
    try:
        await asyncio.wait_for(finished, timeout=timeout_s)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    return messages


@pytest.mark.asyncio
async def test_stream_emits_initial_registry_and_results(paths):
    _write_registry(
        paths["registry"],
        {
            "concept2.alpha": {
                "use_case": "concept2", "pattern_name": "alpha",
                "runs": 2, "scores": [0.60, 0.70], "avg_score": 0.65,
                "last_score": 0.70, "confidence": 0.62, "status": "silver",
                "last_updated": "2026-04-09T10:00:00",
            }
        },
    )
    _write_runs(
        paths["runs"],
        [
            _run_rec("alpha", "concept2", 0.60, ts=1700000100),
            _run_rec("alpha", "concept2", 0.70, ts=1700000200),
        ],
    )
    app = create_app(
        registry_path=paths["registry"],
        policy_path=paths["policy"],
        heartbeat_interval_s=1.0,
        use_cases_dir=paths["use_cases"],
        runs_path=paths["runs"],
    )

    events = await _collect_sse(app, min_events=2, wanted_types={"registry", "result"})

    assert events[0]["event"] == "registry"
    assert events[0]["data"]["patterns"][0]["pattern"] == "concept2.alpha"
    result_events = [ev for ev in events if ev["event"] == "result"]
    assert result_events, "expected initial result events from runs.json"
    assert result_events[0]["data"]["pattern"] == "concept2.alpha"
    assert "commit" in result_events[0]["data"]
    assert "dataset_id" in result_events[0]["data"]


@pytest.mark.asyncio
async def test_stream_emits_incremental_result_on_runs_append(paths):
    _write_runs(
        paths["runs"],
        [_run_rec("alpha", "concept2", 0.60, ts=1700000100)],
    )
    app = create_app(
        registry_path=paths["registry"],
        policy_path=paths["policy"],
        heartbeat_interval_s=1.0,
        use_cases_dir=paths["use_cases"],
        runs_path=paths["runs"],
    )

    async def append_run():
        await asyncio.sleep(0.7)
        existing = json.loads(paths["runs"].read_text())
        existing.append(_run_rec("alpha", "concept2", 0.75, ts=1700000200))
        _write_runs(paths["runs"], existing)

    grow_task = asyncio.create_task(append_run())
    try:
        events = await _collect_sse(app, min_events=2, wanted_types={"result"})
    finally:
        await grow_task

    assert any(ev["data"]["score"] == pytest.approx(0.75) for ev in events)
    assert any(ev["data"]["ts"] == 1700000200 for ev in events)


@pytest.mark.asyncio
async def test_stream_emits_heartbeat(paths):
    app = create_app(
        registry_path=paths["registry"],
        policy_path=paths["policy"],
        heartbeat_interval_s=0.2,
        use_cases_dir=paths["use_cases"],
        runs_path=paths["runs"],
    )
    events = await _collect_sse(app, min_events=1, wanted_types={"heartbeat"}, timeout_s=3.0)
    assert events, "expected at least one heartbeat event"
    assert events[0]["event"] == "heartbeat"
    assert "ts" in events[0]["data"]
