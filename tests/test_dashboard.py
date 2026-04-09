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
    """Create a minimal use_cases/fraud/patterns/ tree matching real pattern names."""
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


# ---------------------------------------------------------------------------
# Basic endpoint tests
# ---------------------------------------------------------------------------


def test_root_returns_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "AIR LAB OS" in r.text


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
    # sorted descending by confidence → logistic first
    assert body["patterns"][0]["pattern"] == "logistic"
    assert body["patterns"][0]["status"] == "gold"
    assert body["patterns"][1]["pattern"] == "velocity"
    assert body["total_experiments"] == 7


def test_registry_sorted_by_tier(client, paths):
    """Gold → silver → bronze regardless of confidence value."""
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
    # Even though bronze_high has a higher confidence, tier order wins.
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
    assert body["total"] == 5  # 3 + 2
    assert body["returned"] == 2
    assert len(body["rows"]) == 2
    # Most recent first: logistic's latest score (0.82) should lead.
    assert body["rows"][0]["pattern"] == "logistic"
    assert body["rows"][0]["score"] == pytest.approx(0.82)


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


# ---------------------------------------------------------------------------
# SSE tests
# ---------------------------------------------------------------------------


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
    """
    Drive the ASGI app directly (bypassing httpx's ASGITransport which
    buffers the full response) and collect SSE events until min_events
    are received, all wanted_types have appeared, or timeout elapses.
    """
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/api/stream",
        "raw_path": b"/api/stream",
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }

    events: list[tuple[str, str]] = []
    seen: set[str] = set()
    buf = bytearray()
    done = asyncio.Event()
    disconnect = asyncio.Event()

    async def receive():
        # Block until the test is done, then signal disconnect.
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        t = message.get("type")
        if t == "http.response.body":
            buf.extend(message.get("body", b""))
            while b"\n\n" in buf:
                idx = buf.index(b"\n\n")
                block = bytes(buf[:idx])
                del buf[: idx + 2]
                ev_type, data = _parse_sse_block(block)
                if ev_type is None:
                    continue
                events.append((ev_type, data))
                seen.add(ev_type)
            if len(events) >= min_events and (
                wanted_types is None or wanted_types.issubset(seen)
            ):
                done.set()

    task = asyncio.create_task(app(scope, receive, send))
    try:
        await asyncio.wait_for(done.wait(), timeout=timeout_s)
    except asyncio.TimeoutError:
        pass
    finally:
        disconnect.set()
        task.cancel()
        try:
            await task
        except BaseException:
            pass
    return events


def test_stream_emits_registry_on_connect(paths):
    _write_registry(
        paths["registry"],
        {
            "velocity": {
                "runs": 1,
                "scores": [0.5],
                "avg_score": 0.5,
                "last_score": 0.5,
                "confidence": 0.5,
                "status": "bronze",
                "last_updated": "2026-04-09T10:00:00",
            }
        },
    )
    app = create_app(
        registry_path=paths["registry"],
        policy_path=paths["policy"],
        heartbeat_interval_s=0.3,
        use_cases_dir=paths["use_cases"],
    )
    events = asyncio.run(
        _collect_sse(app, min_events=1, wanted_types={"registry"}, timeout_s=3.0)
    )
    assert events, "no SSE events received"
    # First event must be a registry event.
    assert events[0][0] == "registry"
    payload = json.loads(events[0][1])
    assert payload["patterns"][0]["pattern"] == "velocity"


def test_stream_emits_heartbeat(paths):
    app = create_app(
        registry_path=paths["registry"],
        policy_path=paths["policy"],
        heartbeat_interval_s=0.3,
        use_cases_dir=paths["use_cases"],
    )
    events = asyncio.run(
        _collect_sse(app, min_events=1, wanted_types={"heartbeat"}, timeout_s=3.0)
    )
    types = {ev[0] for ev in events}
    assert "heartbeat" in types
