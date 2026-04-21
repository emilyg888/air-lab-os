SPEC.md — Phase 3: Dashboard + SSE Log Stream

For Claude Code. Read this in full before writing any code. Build in the order listed.

Goal
A read-only web UI that shows live registry state and streams experiment logs as they happen. No writes, no authentication, no new ML code. The server is a thin layer over the existing PatternRegistry and results.tsv — it surfaces data already produced by Phase 1/2.

New dependency
Add exactly one new dependency to pyproject.toml:
toml"fastapi>=0.110",
"uvicorn>=0.29",
No other additions. Jinja2 is bundled with FastAPI's optional extras — do not add it separately. Serve the frontend as a single embedded HTML string from Python (no template files, no static/ directory).

Repo additions
fraud-engine/
src/
dashboard/
**init**.py
server.py ← FastAPI app + SSE endpoint
ui.py ← HTML/CSS/JS as a Python string constant (UI_HTML)
tests/
test_dashboard.py ← HTTP + SSE integration tests
No other files are added or modified except pyproject.toml.

Component 1: src/dashboard/server.py
Routes
MethodPathDescriptionGET /—Serve UI_HTML from ui.pyGET /api/registryJSONCurrent registry state (read fresh from registry.json each call). Accepts ?use_case=<slug> to filter patterns to one use case.GET /api/policyJSONContents of scoring_policy.yaml, parsedGET /api/resultsJSONLast N scores across all registry entries (query params ?n=50, default 50, max 500; ?use_case=<slug> optional)GET /api/use_casesJSON{"use_cases": [{slug, label}, ...]} — union of use_cases/*/ subdirectories and use-case slugs appearing in registry.json keys.GET /api/streamSSELive experiment log stream — see below. Accepts ?use_case=<slug>.

Registry keys are qualified as `<use_case>.<pattern_name>` (e.g. `concept2.erg_load_threshold`). Filtering splits the key on the first `.`. Payloads include `pattern` (qualified), `pattern_name` (short), and `use_case`.
Registry endpoint — /api/registry
Calls PatternRegistry.load(registry_path, results_path) fresh on each request (no caching — the TSV is the source of truth). Returns:
json{
"patterns": [
{
"pattern": "velocity",
"status": "silver",
"confidence": 0.7312,
"runs": 4,
"best_f1": 0.71,
"best_precision": 0.74,
"best_recall": 0.68,
"promotion_candidate": false,
"last_commit": "a3f9c12",
"last_updated": 1718000000
}
],
"promotion_candidates": ["logistic"],
"total_experiments": 12,
"last_updated": 1718000042
}
Sorted descending by confidence.
Results endpoint — /api/results
Reads results.tsv directly with csv.DictReader. Returns rows as JSON array, most-recent-first (reverse row order). Respects ?n= param.
json{
"rows": [...],
"total": 12,
"returned": 12
}
SSE endpoint — /api/stream
This is the core of Phase 3. It tails results.tsv and emits a Server-Sent Event every time a new row is appended by run_experiment().
Implementation contract:

Uses asyncio + aiofiles (add aiofiles>=23.0 to dependencies) to tail the file without blocking.
On connect: emit a registry event with current registry snapshot (same payload as /api/registry), then emit the last 10 result events from existing TSV rows so the client has recent history without a separate HTTP call.
While connected: poll the file every 500ms. When new rows appear, emit one result event per row.
Event types:

event: registry
data: {"patterns": [...], ...}

event: result
data: {"commit": "...", "pattern": "velocity", "score": 0.73, "status": "keep", ...}

event: heartbeat
data: {"ts": 1718000000}

Emit a heartbeat event every 15 seconds to keep proxies from closing the connection.
If results.tsv does not exist yet, keep polling until it appears — do not crash.

Server startup
python# src/dashboard/server.py — bottom of file

if **name** == "**main**":
import uvicorn, argparse
parser = argparse.ArgumentParser()
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--registry", default="registry.json")
parser.add_argument("--results", default="results.tsv")
parser.add_argument("--policy", default="scoring_policy.yaml")
args = parser.parse_args() # Inject paths as app state before serving
uvicorn.run(app, host=args.host, port=args.port)
Run with:
bashuv run python -m src.dashboard.server --port 8000

Component 2: src/dashboard/ui.py
A single Python file exporting UI_HTML: str — the complete frontend as one self-contained HTML string. No external CDN dependencies (all JS inline). No build step.
Aesthetic direction
Industrial terminal + live telemetry. Think: a Bloomberg terminal crossed with a NASA mission control panel. Dark background (#0a0a0f), monospace everywhere (JetBrains Mono via Google Fonts is allowed — one CDN font import only), amber/green accent palette. Status badges are colored chips. The log stream feels like stdout scrolling live.
This is a tool for researchers who live in terminals. It should feel dense, precise, and fast — not a marketing page.
Layout
┌─────────────────────────────────────────────────────────┐
│ FRAUD ENGINE ● LIVE [policy badge] [ts] │ ← header bar
├───────────────────────┬─────────────────────────────────┤
│ REGISTRY │ EXPERIMENT LOG │
│ ┌──────────────────┐ │ ┌─────────────────────────────┐│
│ │ velocity SILVER │ │ │ 14:32:01 velocity 0.7312 ✓ ││
│ │ conf 0.73 4 runs│ │ │ 14:31:45 logistic 0.6891 ✓ ││
│ │ f1=0.71 p=0.74 │ │ │ 14:30:12 velocity 0.6201 ✗ ││
│ └──────────────────┘ │ │ ... ││
│ ┌──────────────────┐ │ └─────────────────────────────┘│
│ │ logistic GOLD ★ │ │ │
│ └──────────────────┘ │ CONNECTION: ● CONNECTED │
├───────────────────────┴─────────────────────────────────┤
│ SCORING POLICY precision_recall=0.40 expl=0.25 ... │ ← footer bar
└─────────────────────────────────────────────────────────┘
Behaviour

On load: fetch /api/registry and /api/results?n=20 simultaneously via Promise.all, populate both panels.
Open /api/stream with EventSource. On result event: prepend a new row to the log panel (max 200 rows retained in DOM). On registry event: re-render the registry panel entirely. On heartbeat: update the "last heartbeat" timestamp in the footer.
Connection status indicator: green dot "CONNECTED" when EventSource is open, amber "RECONNECTING" on error (EventSource auto-reconnects), red "DISCONNECTED" if >30s since last heartbeat.
Status badges: bronze = #8B6914 bg, silver = #5C7A8C bg, gold = #B8860B bg with ★ prefix, crash = #8C2020 bg.
Score bar: each registry card shows a horizontal bar filling proportionally to confidence (0→1). Silver threshold (0.65) and gold threshold (0.78) shown as vertical tick marks on the bar.
Promotion candidates: registry cards with promotion_candidate: true pulse with a slow amber glow animation. A banner reads ⚑ PROMOTION CANDIDATE — manual review required.
Log rows: keep rows have a ✓ in green, discard rows have ✗ in dim amber, crash rows have ⚡ in red. Clicking a log row expands an inline detail panel showing all fields.
No JavaScript frameworks. Vanilla JS only. The JS section should be under ~200 lines.

Component 3: tests/test_dashboard.py
pythonimport pytest
import threading
import time
from pathlib import Path
from fastapi.testclient import TestClient
Use FastAPI's TestClient (wraps requests, no running server needed). Patch the file paths via app state.
Tests to include:
TestWhat it checkstest_root_returns_htmlGET / returns 200 and Content-Type: text/htmltest_registry_empty/api/registry returns valid JSON with empty patterns list when TSV doesn't existtest_registry_reflects_tsvWrite a known TSV row, hit /api/registry, assert pattern appears with correct confidencetest_results_endpoint/api/results returns rows in reverse order, ?n=2 limits correctlytest_policy_endpoint/api/policy returns parsed YAML with weights key presenttest_stream_emits_registry_on_connectSSE stream first event is type registrytest_stream_emits_heartbeatSSE stream emits heartbeat within 20s (use TestClient with stream=True, timeout)
SSE tests: use TestClient with stream=True and read the raw response bytes line-by-line. Parse event: and data: lines manually — no SSE client library.

Rules this phase must not break
All rules from CLAUDE.md carry forward:

scoring_policy.yaml — read only. The /api/policy endpoint reads it; never writes it.
results.tsv — the SSE tail reads it; the server never writes it.
registry.json — the server reads it for gold status preservation only; never writes it.
uv run pytest tests/ -q must pass all tests including the new dashboard tests.

Deliverables checklist

uv run pytest tests/ -q — zero failures across all test files
uv run python -m src.dashboard.server --port 8000 starts without error
GET / returns the full HTML UI (visible in browser)
GET /api/registry returns correct JSON after running two experiments
GET /api/stream streams events — verify with curl -N http://localhost:8000/api/stream
Running an experiment in a second terminal while the dashboard is open causes a new row to appear in the log panel within 1 second
scoring_policy.yaml unchanged (weights still sum to 1.0)
CLAUDE.md unchanged

What Claude Code must NOT do in Phase 3

No writes to results.tsv, registry.json, or scoring_policy.yaml from the server
No authentication, sessions, or user accounts
No WebSocket (SSE only — simpler, no upgrade negotiation, reconnect is free)
No React, Vue, Svelte, or any JS framework — vanilla JS only
No Jinja2 templates — the HTML is a Python string constant in ui.py
No static/ directory — everything in one UI_HTML string
No background threads in the server process — use asyncio for the SSE tail
No new ML dependencies

Key design decisions (rationale for Claude Code)
Why SSE not WebSocket? SSE is one-directional (server→client), which matches the constraint that the dashboard is read-only. It reconnects automatically, works through HTTP/1.1 proxies, and requires zero client-side handshake code.
Why tail results.tsv not poll /api/registry? The TSV is append-only and line-oriented — tailing it is a natural fit. Polling the registry would require re-parsing the entire file on every poll. Tailing gives us row-level events cheaply.
Why rebuild PatternRegistry on every /api/registry request? The registry is deliberately derived state (CLAUDE.md rule 2). Caching it would let the dashboard show stale data during an active experiment run. The rebuild is fast (linear scan of a small TSV).
Why embed UI as a Python string? Zero deployment friction. uv run python -m src.dashboard.server is the entire deployment. No asset pipeline, no CDN dependency for JS, no MANIFEST.in.

End of Phase 3 spec. Next: Phase 4 — Auto-promote loop + notifications.

A few things worth flagging before Claude Code starts building:
The aiofiles dependency is the only addition beyond FastAPI + uvicorn. Without it, the SSE tail blocks the event loop. Make sure it goes into pyproject.toml before anything else.
SSE test complexity: the test_stream_emits_heartbeat test needs a real 15-second wait unless you make the heartbeat interval configurable (recommended — add a heartbeat_interval_s param to the app, default 15, set to 1 in tests).
Concurrent experiment + dashboard: the SSE tail opens results.tsv for reading while run_experiment() is appending. On Linux this is fine (POSIX file semantics). On Windows it can lock — worth a # NOTE: Windows file locking comment in the tail code.
