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


def _make_use_cases_tree(tmp_path: Path) -> Path:
    root = tmp_path / "use_cases"
    fraud = root / "fraud" / "patterns"
    fraud.mkdir(parents=True)
    for name in ("rule_velocity.py", "ml_logistic.py", "rule_spike.py"):
        (fraud / name).write_text("")
    (fraud / "__init__.py").write_text("")
    return root


@pytest.fixture
def paths(tmp_path):
    return {
        "registry": tmp_path / "registry.json",
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
    )
    return TestClient(app)


def test_root_returns_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "AIR LAB OS" in r.text
    assert 'id="detail-modal"' in r.text
    assert 'id="log-title"' in r.text
    assert 'id="feature-strip"' in r.text


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


def test_results_endpoint_limits_and_order(client, paths):
    _write_registry(
        paths["registry"],
        {
            "velocity": {
                "runs": 3,
                "scores": [0.60, 0.65, 0.70],
                "avg_score": 0.65,
                "last_score": 0.70,
                "confidence": 0.60,
                "status": "silver",
                "last_updated": "2026-04-09T10:00:00",
            },
            "logistic": {
                "runs": 2,
                "scores": [0.80, 0.82],
                "avg_score": 0.81,
                "last_score": 0.82,
                "confidence": 0.78,
                "status": "gold",
                "last_updated": "2026-04-09T11:00:00",
            },
        },
    )
    r = client.get("/api/results?n=2")
    body = r.json()
    assert body["total"] == 5
    assert body["returned"] == 2
    assert len(body["rows"]) == 2
    assert body["rows"][0]["pattern"] == "logistic"
    assert body["rows"][0]["score"] == pytest.approx(0.82)
    assert body["rows"][1]["pattern"] == "logistic"
    assert body["rows"][1]["score"] == pytest.approx(0.80)


def test_registry_exposes_use_case(client, paths):
    _write_registry(
        paths["registry"],
        {
            "rule_velocity": {
                "runs": 2,
                "scores": [0.6, 0.7],
                "avg_score": 0.65,
                "last_score": 0.7,
                "confidence": 0.6,
                "status": "silver",
                "last_updated": "2026-04-09T10:00:00",
            },
            "ml_logistic": {
                "runs": 3,
                "scores": [0.9, 0.91, 0.92],
                "avg_score": 0.91,
                "last_score": 0.92,
                "confidence": 0.9,
                "status": "gold",
                "is_stable": True,
                "last_updated": "2026-04-09T11:00:00",
            },
        },
    )
    body = client.get("/api/registry").json()
    assert "use_case" in body
    assert body["use_case"]["slug"] == "fraud"
    assert body["use_case"]["current"] == "Fraud Detection"
    assert body["use_case"]["unmatched"] == []


def test_policy_endpoint(client):
    r = client.get("/api/policy")
    assert r.status_code == 200
    body = r.json()
    assert "weights" in body
    assert body["weights"]["primary_metric"] == 0.40
    assert body["promotion"]["working_threshold"] == 0.65


def test_features_endpoint_returns_snapshot(client):
    r = client.get("/api/features")
    assert r.status_code == 200
    body = r.json()
    assert body["dataset"] == "use_cases.fraud.handle"
    assert body["baseline_f1"] == pytest.approx(0.8671)
    assert body["counts"] == {"improved": 0, "flat": 2, "regressed": 5}
    assert len(body["results"]) == 7
    assert body["results"][0]["feature_name"] == "amount_to_balance_ratio"


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
            "alpha": {
                "runs": 2,
                "scores": [0.60, 0.70],
                "avg_score": 0.65,
                "last_score": 0.70,
                "confidence": 0.62,
                "status": "silver",
                "last_updated": "2026-04-09T10:00:00",
            }
        },
    )
    app = create_app(
        registry_path=paths["registry"],
        policy_path=paths["policy"],
        heartbeat_interval_s=1.0,
        use_cases_dir=paths["use_cases"],
    )

    events = await _collect_sse(app, min_events=2, wanted_types={"registry", "result"})

    assert events[0]["event"] == "registry"
    assert events[0]["data"]["patterns"][0]["pattern"] == "alpha"
    assert any(ev["event"] == "result" for ev in events)


@pytest.mark.asyncio
async def test_stream_emits_incremental_result_on_registry_growth(paths):
    _write_registry(
        paths["registry"],
        {
            "alpha": {
                "runs": 1,
                "scores": [0.60],
                "avg_score": 0.60,
                "last_score": 0.60,
                "confidence": 0.60,
                "status": "bronze",
                "last_updated": "2026-04-09T10:00:00",
            }
        },
    )
    app = create_app(
        registry_path=paths["registry"],
        policy_path=paths["policy"],
        heartbeat_interval_s=1.0,
        use_cases_dir=paths["use_cases"],
    )

    async def grow_registry():
        await asyncio.sleep(0.7)
        _write_registry(
            paths["registry"],
            {
                "alpha": {
                    "runs": 2,
                    "scores": [0.60, 0.75],
                    "avg_score": 0.675,
                    "last_score": 0.75,
                    "confidence": 0.64,
                    "status": "silver",
                    "last_updated": "2026-04-09T10:05:00",
                }
            },
        )

    grow_task = asyncio.create_task(grow_registry())
    try:
        events = await _collect_sse(app, min_events=3, wanted_types={"registry", "result"})
    finally:
        await grow_task

    result_events = [ev for ev in events if ev["event"] == "result"]
    assert any(ev["data"]["run_index"] == 1 for ev in result_events)
    assert any(ev["data"]["score"] == pytest.approx(0.75) for ev in result_events)
