"""Embedded HTML/CSS/JS for the dashboard UI.

Single self-contained string. No build step, no framework, vanilla JS only.
"""

UI_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>AIR LAB OS — mission control</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0a0f;
    --bg-soft: #11121a;
    --bg-card: #151723;
    --border: #1f2233;
    --text: #d7d8e0;
    --text-dim: #6a6d82;
    --amber: #f5b342;
    --green: #4ade80;
    --red: #ef4444;
    --bronze: #8B6914;
    --silver: #5C7A8C;
    --gold: #B8860B;
    --crash: #8C2020;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    margin: 0;
    font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 12px;
    background: var(--bg);
    color: var(--text);
    display: flex;
    flex-direction: column;
  }
  header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    background: var(--bg-soft);
    flex-shrink: 0;
  }
  header .brand {
    font-weight: 700;
    letter-spacing: 2px;
    color: var(--amber);
  }
  header .live {
    color: var(--green);
    font-weight: 500;
  }
  header .dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 8px var(--green);
    animation: pulse 2s ease-in-out infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }
  header .spacer { flex: 1; }
  header .meta { color: var(--text-dim); font-size: 11px; }
  header .policy-badge {
    padding: 3px 8px;
    border: 1px solid var(--border);
    border-radius: 2px;
    background: var(--bg-card);
    color: var(--amber);
    font-size: 10px;
    letter-spacing: 1px;
  }
  header .use-case {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 10px;
    border: 1px solid var(--border);
    border-radius: 2px;
    background: var(--bg-card);
    font-size: 11px;
    letter-spacing: 1px;
  }
  header .use-case .label {
    color: var(--text-dim);
    text-transform: uppercase;
    font-size: 10px;
  }
  header .use-case select {
    background: var(--bg-card);
    color: var(--green);
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 2px 6px;
    font-family: inherit;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    cursor: pointer;
  }
  header .use-case select:focus { outline: 1px solid var(--amber); }

  main {
    display: grid;
    grid-template-columns: minmax(340px, 1fr) 1.4fr;
    gap: 0;
    flex: 1;
    min-height: 0;
  }
  .panel {
    display: flex;
    flex-direction: column;
    min-height: 0;
    border-right: 1px solid var(--border);
  }
  .panel:last-child { border-right: none; }
  .panel-title {
    padding: 8px 16px;
    font-size: 10px;
    letter-spacing: 2px;
    color: var(--text-dim);
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    background: var(--bg-soft);
    flex-shrink: 0;
  }
  .panel-title.dynamic::after {
    content: attr(data-view-label);
    color: var(--amber);
    margin-left: 10px;
  }
  .panel-body { flex: 1; overflow-y: auto; padding: 10px 12px; }
  .panel-body::-webkit-scrollbar { width: 8px; }
  .panel-body::-webkit-scrollbar-thumb { background: var(--border); }

  .registry-toolbar {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 12px;
  }
  .registry-filter {
    border: 1px solid var(--border);
    background: var(--bg-soft);
    color: var(--text-dim);
    padding: 6px 10px;
    font: inherit;
    cursor: pointer;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-size: 10px;
  }
  .registry-filter:hover {
    color: var(--text);
    border-color: rgba(245, 179, 66, 0.45);
  }
  .registry-filter.active {
    color: var(--amber);
    border-color: var(--amber);
    background: rgba(245, 179, 66, 0.08);
  }

  .banner {
    display: none;
    margin: 0 12px 10px 12px;
    padding: 8px 12px;
    border: 1px solid var(--amber);
    background: rgba(245, 179, 66, 0.08);
    color: var(--amber);
    letter-spacing: 1px;
    font-weight: 500;
    animation: glow 2.5s ease-in-out infinite;
  }
  .banner.active { display: block; }
  @keyframes glow {
    0%, 100% { box-shadow: 0 0 0 rgba(245, 179, 66, 0); }
    50% { box-shadow: 0 0 12px rgba(245, 179, 66, 0.35); }
  }

  .pattern-card {
    border: 1px solid var(--border);
    background: var(--bg-card);
    padding: 10px 12px;
    margin-bottom: 8px;
    position: relative;
  }
  .pattern-card.promotion {
    animation: candidate-glow 3s ease-in-out infinite;
    border-color: var(--amber);
  }
  @keyframes candidate-glow {
    0%, 100% { box-shadow: 0 0 0 rgba(245, 179, 66, 0); }
    50% { box-shadow: 0 0 14px rgba(245, 179, 66, 0.4); }
  }
  .pattern-card .row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
  }
  .pattern-card .name {
    font-weight: 700;
    color: var(--text);
  }
  .badge {
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 2px;
    letter-spacing: 1px;
    color: #fff;
    text-transform: uppercase;
    font-weight: 700;
  }
  .badge.bronze { background: var(--bronze); }
  .badge.silver { background: var(--silver); }
  .badge.gold   { background: var(--gold); }
  .badge.crash  { background: var(--crash); }

  .meta-line {
    margin-top: 4px;
    color: var(--text-dim);
    font-size: 11px;
  }

  .score-bar {
    position: relative;
    margin-top: 8px;
    height: 6px;
    background: #0b0c13;
    border: 1px solid var(--border);
  }
  .score-bar .fill {
    position: absolute;
    top: 0; left: 0; bottom: 0;
    background: linear-gradient(90deg, var(--amber), var(--green));
  }
  .score-bar .tick {
    position: absolute;
    top: -2px;
    bottom: -2px;
    width: 1px;
    background: var(--text-dim);
  }

  .log-row {
    display: grid;
    grid-template-columns: 70px 1fr 70px 20px;
    gap: 10px;
    padding: 4px 10px;
    border-bottom: 1px solid rgba(31, 34, 51, 0.5);
    cursor: pointer;
    font-size: 12px;
  }
  .log-row:hover { background: rgba(31, 34, 51, 0.5); }
  .log-row .ts { color: var(--text-dim); }
  .log-row .pat { color: var(--text); }
  .log-row .score { color: var(--amber); text-align: right; }
  .log-row .marker {
    text-align: center;
    font-size: 10px;
    letter-spacing: 1px;
    text-transform: uppercase;
  }
  .log-row .marker.gold { color: var(--gold); }
  .log-row .marker.silver { color: var(--silver); }
  .log-row .marker.bronze { color: var(--bronze); }
  .log-row .marker.keep { color: var(--green); }
  .log-row .marker.discard { color: #8a6a1f; opacity: 0.7; }
  .log-row .marker.crash { color: var(--red); }
  .log-row .score-label {
    color: var(--text-dim);
    margin-right: 4px;
  }

  .detail-modal[hidden] { display: none; }
  .detail-modal {
    position: fixed;
    inset: 0;
    z-index: 20;
    display: grid;
    place-items: center;
    padding: 20px;
    background: rgba(5, 6, 10, 0.78);
    backdrop-filter: blur(4px);
  }
  .detail-card {
    width: min(680px, 100%);
    max-height: min(80vh, 720px);
    overflow: auto;
    border: 1px solid var(--border);
    background: linear-gradient(180deg, rgba(21, 23, 35, 0.98), rgba(12, 14, 23, 0.98));
    box-shadow: 0 18px 48px rgba(0, 0, 0, 0.45);
  }
  .detail-head {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 16px;
    padding: 16px;
    border-bottom: 1px solid var(--border);
  }
  .detail-kicker {
    color: var(--text-dim);
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
  }
  .detail-title {
    margin-top: 6px;
    color: var(--text);
    font-size: 16px;
    font-weight: 700;
  }
  .detail-subtitle {
    margin-top: 6px;
    color: var(--text-dim);
    font-size: 11px;
    line-height: 1.6;
  }
  .detail-close {
    border: 1px solid var(--border);
    background: transparent;
    color: var(--text-dim);
    font: inherit;
    cursor: pointer;
    padding: 6px 10px;
  }
  .detail-close:hover { color: var(--amber); border-color: var(--amber); }
  .detail-body {
    padding: 16px;
  }
  .detail-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
    margin-bottom: 14px;
  }
  .detail-stat {
    border: 1px solid var(--border);
    background: rgba(10, 10, 15, 0.6);
    padding: 10px;
  }
  .detail-stat-label {
    color: var(--text-dim);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  .detail-stat-value {
    margin-top: 6px;
    color: var(--text);
    font-size: 14px;
  }
  .detail-json-label {
    color: var(--text-dim);
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
  }
  .detail-json {
    margin: 0;
    white-space: pre-wrap;
    color: var(--text-dim);
    border: 1px solid var(--border);
    background: #090b12;
    padding: 12px;
    overflow-x: auto;
  }

  .connection {
    padding: 8px 16px;
    border-top: 1px solid var(--border);
    background: var(--bg-soft);
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    flex-shrink: 0;
  }
  .connection .dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
  }
  .connection.connected .dot { background: var(--green); box-shadow: 0 0 6px var(--green); }
  .connection.reconnecting .dot { background: var(--amber); }
  .connection.disconnected .dot { background: var(--red); }
  .connection.connected { color: var(--green); }
  .connection.reconnecting { color: var(--amber); }
  .connection.disconnected { color: var(--red); }

  footer {
    padding: 8px 16px;
    border-top: 1px solid var(--border);
    background: var(--bg-soft);
    color: var(--text-dim);
    font-size: 11px;
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    flex-shrink: 0;
  }
  footer strong { color: var(--amber); font-weight: 500; letter-spacing: 1px; }

  .empty {
    padding: 20px;
    color: var(--text-dim);
    text-align: center;
    font-style: italic;
  }

  @media (max-width: 760px) {
    main {
      grid-template-columns: 1fr;
    }
    .panel {
      border-right: none;
      border-bottom: 1px solid var(--border);
    }
    .panel:last-child { border-bottom: none; }
    .log-row {
      grid-template-columns: 64px 1fr 92px 20px;
      font-size: 11px;
    }
    .detail-grid {
      grid-template-columns: 1fr;
    }
  }
</style>
</head>
<body>

<header>
  <span class="brand">AIR LAB OS</span>
  <span class="dot"></span>
  <span class="live">LIVE</span>
  <span class="use-case" id="use-case-box">
    <span class="label">USE CASE</span>
    <select id="use-case-select" aria-label="Use case">
      <option value="all">ALL</option>
    </select>
  </span>
  <span class="spacer"></span>
  <span class="policy-badge" id="policy-badge">POLICY —</span>
  <span class="meta" id="header-ts">—</span>
</header>

<div id="promotion-banner" class="banner">⚑ PROMOTION CANDIDATE — manual review required</div>

<main>
  <section class="panel">
    <div class="panel-title">REGISTRY</div>
    <div class="panel-body" id="registry-panel">
      <div class="empty">awaiting first snapshot…</div>
    </div>
  </section>
  <section class="panel">
    <div class="panel-title dynamic" id="log-title" data-view-label="· REGISTRY">EXPERIMENT LOG</div>
    <div class="panel-body" id="log-panel">
      <div class="empty">awaiting events…</div>
    </div>
    <div class="connection disconnected" id="conn">
      <span class="dot"></span>
      <span id="conn-label">CONNECTING…</span>
    </div>
  </section>
</main>

<div class="detail-modal" id="detail-modal" hidden>
  <div class="detail-card" role="dialog" aria-modal="true" aria-labelledby="detail-title">
    <div class="detail-head">
      <div>
        <div class="detail-kicker">Run Details</div>
        <div class="detail-title" id="detail-title">—</div>
        <div class="detail-subtitle" id="detail-subtitle">—</div>
      </div>
      <button class="detail-close" id="detail-close" type="button">Close</button>
    </div>
    <div class="detail-body">
      <div class="detail-grid">
        <div class="detail-stat">
          <div class="detail-stat-label">Metric Score</div>
          <div class="detail-stat-value" id="detail-score">—</div>
        </div>
        <div class="detail-stat">
          <div class="detail-stat-label">Status</div>
          <div class="detail-stat-value" id="detail-status">—</div>
        </div>
        <div class="detail-stat">
          <div class="detail-stat-label">Run Index</div>
          <div class="detail-stat-value" id="detail-run-index">—</div>
        </div>
        <div class="detail-stat">
          <div class="detail-stat-label">Timestamp</div>
          <div class="detail-stat-value" id="detail-ts">—</div>
        </div>
      </div>
      <div class="detail-json-label">Event Payload</div>
      <pre class="detail-json" id="detail-json">—</pre>
    </div>
  </div>
</div>

<footer>
  <span><strong>SCORING POLICY</strong> <span id="policy-weights">—</span></span>
  <span><strong>LAST HEARTBEAT</strong> <span id="heartbeat-ts">never</span></span>
</footer>

<script>
(() => {
  const MAX_LOG_ROWS = 200;
  const WORKING = 0.65;
  const STABLE  = 0.78;

  const registryPanel = document.getElementById('registry-panel');
  const logPanel      = document.getElementById('log-panel');
  const policyBadge   = document.getElementById('policy-badge');
  const policyWeights = document.getElementById('policy-weights');
  const headerTs      = document.getElementById('header-ts');
  const useCaseSelect = document.getElementById('use-case-select');
  const heartbeatTs   = document.getElementById('heartbeat-ts');
  const conn          = document.getElementById('conn');
  const connLabel     = document.getElementById('conn-label');
  const banner        = document.getElementById('promotion-banner');
  const logTitle      = document.getElementById('log-title');
  const detailModal   = document.getElementById('detail-modal');
  const detailClose   = document.getElementById('detail-close');
  const detailTitle   = document.getElementById('detail-title');
  const detailSubtitle = document.getElementById('detail-subtitle');
  const detailScore   = document.getElementById('detail-score');
  const detailStatus  = document.getElementById('detail-status');
  const detailRunIndex = document.getElementById('detail-run-index');
  const detailTs      = document.getElementById('detail-ts');
  const detailJson    = document.getElementById('detail-json');

  let lastHeartbeat = 0;
  let thresholds = { working: WORKING, stable: STABLE };
  let selectedLogView = 'registry';
  let patternStatusByName = {};
  let logEvents = [];
  const seenLogEvents = new Set();

  const fmtTs = (s) => {
    if (!s) return '—';
    if (typeof s === 'number') s = new Date(s * 1000).toISOString();
    return String(s).slice(11, 19) || String(s).slice(0, 19);
  };

  const escapeHtml = (str) =>
    String(str).replace(/[&<>"']/g, (c) =>
      ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  const parseEventTime = (value) => {
    if (typeof value === 'number' && Number.isFinite(value)) return value;
    if (typeof value === 'string' && value) {
      const asNumber = Number(value);
      if (Number.isFinite(asNumber) && value.trim() !== '') return asNumber;
      const parsed = Date.parse(value);
      if (Number.isFinite(parsed)) return parsed / 1000;
    }
    return 0;
  };

  const normalizeLogEvent = (ev) => ({
    pattern: String(ev?.pattern || '?'),
    run_index: Number.isFinite(Number(ev?.run_index)) ? Number(ev.run_index) : -1,
    score: Number.isFinite(Number(ev?.score)) ? Number(ev.score) : 0,
    status: String(ev?.status || 'keep'),
    last_updated: ev?.last_updated ?? '',
    ts: ev?.ts ?? ev?.last_updated ?? '',
    is_last: Boolean(ev?.is_last),
  });

  const logEventKey = (ev) =>
    [
      ev.pattern,
      ev.run_index,
      ev.score.toFixed(8),
      ev.status,
      String(ev.last_updated || ''),
    ].join('|');

  const compareLogEvents = (a, b) => {
    if (a.score !== b.score) return b.score - a.score;
    const timeDiff = parseEventTime(b.ts || b.last_updated) - parseEventTime(a.ts || a.last_updated);
    if (timeDiff !== 0) return timeDiff;
    if (a.run_index !== b.run_index) return b.run_index - a.run_index;
    return a.pattern.localeCompare(b.pattern);
  };

  const viewLabel = (view) => {
    if (view === 'gold') return 'GOLD';
    if (view === 'silver') return 'SILVER';
    if (view === 'bronze') return 'BRONZE';
    return 'REGISTRY';
  };

  const eventTier = (ev) => String(patternStatusByName[ev.pattern] || ev.status || 'bronze').toLowerCase();

  function matchesSelectedView(ev) {
    if (selectedLogView === 'registry') return true;
    return eventTier(ev) === selectedLogView;
  }

  function setSelectedLogView(view) {
    selectedLogView = view;
    logTitle.dataset.viewLabel = `· ${viewLabel(view)}`;
    renderRegistryFilters();
    renderLogPanel();
  }

  function renderRegistryFilters(counts = null) {
    const toolbar = registryPanel.querySelector('[data-role="registry-toolbar"]');
    if (!toolbar) return;

    const sourceCounts = counts || {
      registry: Object.keys(patternStatusByName).length,
      gold: Object.values(patternStatusByName).filter((s) => s === 'gold').length,
      silver: Object.values(patternStatusByName).filter((s) => s === 'silver').length,
      bronze: Object.values(patternStatusByName).filter((s) => s === 'bronze').length,
    };

    toolbar.querySelectorAll('.registry-filter').forEach((button) => {
      const view = button.dataset.view || 'registry';
      button.classList.toggle('active', view === selectedLogView);
      const count = sourceCounts[view] ?? 0;
      button.textContent = `${viewLabel(view)} ${count}`;
    });
  }

  function renderLogPanel() {
    const visibleEvents = logEvents.filter(matchesSelectedView);

    if (!logEvents.length) {
      logPanel.innerHTML = '<div class="empty">awaiting events…</div>';
      return;
    }

    if (!visibleEvents.length) {
      logPanel.innerHTML = `<div class="empty">no past runs in ${escapeHtml(viewLabel(selectedLogView).toLowerCase())}</div>`;
      return;
    }

    logPanel.innerHTML = '';
    visibleEvents.forEach((ev) => {
      const tier = eventTier(ev);
      const marker = tierToMarker(tier);
      const ts = fmtTs(ev.ts || ev.last_updated);
      const row = document.createElement('div');
      row.className = 'log-row';
      row.innerHTML = `
        <span class="ts">${escapeHtml(ts)}</span>
        <span class="pat">${escapeHtml(ev.pattern || '?')}</span>
        <span class="score"><span class="score-label">score</span>${Number(ev.score || 0).toFixed(4)}</span>
        <span class="marker ${marker.cls}">${marker.char}</span>
      `;
      row.addEventListener('click', () => openDetailModal(ev));
      logPanel.appendChild(row);
    });
  }

  function renderRegistry(snapshot) {
    if (snapshot.thresholds) thresholds = snapshot.thresholds;
    const patterns = snapshot.patterns || [];
    patternStatusByName = Object.fromEntries(
      patterns.map((p) => [p.pattern, String(p.status || 'bronze').toLowerCase()])
    );
    headerTs.textContent = new Date().toISOString().slice(0, 19) + 'Z';
    if (!patterns.length) {
      registryPanel.innerHTML = `
        <div class="registry-toolbar" data-role="registry-toolbar">
          <button class="registry-filter active" data-view="registry" type="button">REGISTRY 0</button>
          <button class="registry-filter" data-view="gold" type="button">GOLD 0</button>
          <button class="registry-filter" data-view="silver" type="button">SILVER 0</button>
          <button class="registry-filter" data-view="bronze" type="button">BRONZE 0</button>
        </div>
        <div class="empty">no patterns yet</div>
      `;
      bindRegistryFilters();
      banner.classList.remove('active');
      renderLogPanel();
      return;
    }
    const working = thresholds.working || WORKING;
    const stable  = thresholds.stable  || STABLE;
    const counts = {
      registry: patterns.length,
      gold: patterns.filter((p) => String(p.status || '').toLowerCase() === 'gold').length,
      silver: patterns.filter((p) => String(p.status || '').toLowerCase() === 'silver').length,
      bronze: patterns.filter((p) => String(p.status || '').toLowerCase() === 'bronze').length,
    };
    const html = patterns.map((p) => {
      const status = (p.status || 'bronze').toLowerCase();
      const star = status === 'gold' ? '★ ' : '';
      const conf = (p.confidence ?? 0);
      const fill = Math.max(0, Math.min(1, conf)) * 100;
      const candClass = p.promotion_candidate ? ' promotion' : '';
      return `
        <div class="pattern-card${candClass}">
          <div class="row">
            <div class="name">${escapeHtml(p.pattern)}</div>
            <div class="badge ${status}">${star}${status.toUpperCase()}</div>
          </div>
          <div class="meta-line">
            conf ${conf.toFixed(4)} · runs ${p.runs ?? 0} ·
            avg ${(p.avg_score ?? 0).toFixed(4)} ·
            last ${(p.last_score ?? 0).toFixed(4)}
            ${p.is_stable ? '· stable' : ''}
          </div>
          <div class="score-bar">
            <div class="fill" style="width:${fill}%"></div>
            <div class="tick" style="left:${working * 100}%"></div>
            <div class="tick" style="left:${stable * 100}%"></div>
          </div>
        </div>
      `;
    }).join('');
    registryPanel.innerHTML = `
      <div class="registry-toolbar" data-role="registry-toolbar">
        <button class="registry-filter" data-view="registry" type="button">REGISTRY ${counts.registry}</button>
        <button class="registry-filter" data-view="gold" type="button">GOLD ${counts.gold}</button>
        <button class="registry-filter" data-view="silver" type="button">SILVER ${counts.silver}</button>
        <button class="registry-filter" data-view="bronze" type="button">BRONZE ${counts.bronze}</button>
      </div>
      ${html}
    `;
    bindRegistryFilters();
    renderRegistryFilters(counts);
    logEvents = logEvents
      .map((ev) => ({ ...ev, status: eventTier(ev) }))
      .sort(compareLogEvents);
    renderLogPanel();

    const hasCand = (snapshot.promotion_candidates || []).length > 0;
    banner.classList.toggle('active', hasCand);
  }

  function bindRegistryFilters() {
    registryPanel.querySelectorAll('.registry-filter').forEach((button) => {
      button.addEventListener('click', () => setSelectedLogView(button.dataset.view || 'registry'));
    });
  }

  function renderPolicy(policy) {
    if (!policy || !policy.weights) return;
    const w = policy.weights;
    policyBadge.textContent = 'POLICY v' + (policy.version || '?');
    policyWeights.textContent = Object.entries(w)
      .map(([k, v]) => `${k}=${v}`)
      .join('  ·  ');
  }

  function statusToMarker(status) {
    const s = (status || '').toLowerCase();
    if (s === 'crash')   return { cls: 'crash',   char: '⚡' };
    if (s === 'discard') return { cls: 'discard', char: '✗' };
    return { cls: 'keep', char: '✓' };
  }

  function tierToMarker(tier) {
    const normalized = String(tier || 'bronze').toLowerCase();
    if (normalized === 'gold') return { cls: 'gold', char: 'G' };
    if (normalized === 'silver') return { cls: 'silver', char: 'S' };
    return { cls: 'bronze', char: 'B' };
  }

  function addLogEvent(ev) {
    const normalized = normalizeLogEvent(ev);
    const key = logEventKey(normalized);
    if (seenLogEvents.has(key)) return;

    seenLogEvents.add(key);
    logEvents.push(normalized);
    logEvents.sort(compareLogEvents);

    while (logEvents.length > MAX_LOG_ROWS) {
      const removed = logEvents.pop();
      if (removed) seenLogEvents.delete(logEventKey(removed));
    }

    renderLogPanel();
  }

  function openDetailModal(ev) {
    detailTitle.textContent = ev.pattern || 'Unknown pattern';
    detailSubtitle.textContent = [
      `${viewLabel(eventTier(ev))} tier`,
      `latest flag: ${ev.is_last ? 'yes' : 'no'}`,
    ].join(' · ');
    detailScore.textContent = Number(ev.score || 0).toFixed(4);
    detailStatus.textContent = viewLabel(eventTier(ev));
    detailRunIndex.textContent = ev.run_index >= 0 ? String(ev.run_index + 1) : '—';
    detailTs.textContent = String(ev.ts || ev.last_updated || '—');
    detailJson.textContent = JSON.stringify(ev, null, 2);
    detailModal.hidden = false;
    document.body.style.overflow = 'hidden';
  }

  function closeDetailModal() {
    detailModal.hidden = true;
    document.body.style.overflow = '';
  }

  function setConn(state, label) {
    conn.classList.remove('connected', 'reconnecting', 'disconnected');
    conn.classList.add(state);
    connLabel.textContent = label;
  }

  let currentEventSource = null;

  function selectedUseCase() {
    const v = useCaseSelect ? useCaseSelect.value : 'all';
    return (!v || v === 'all') ? '' : v;
  }

  function useCaseQuery() {
    const slug = selectedUseCase();
    return slug ? ('&use_case=' + encodeURIComponent(slug)) : '';
  }

  async function populateUseCases() {
    try {
      const payload = await fetch('/api/use_cases').then(r => r.json());
      const list = (payload && Array.isArray(payload.use_cases)) ? payload.use_cases : [];
      const existing = useCaseSelect.value;
      useCaseSelect.innerHTML = '<option value="all">ALL</option>'
        + list.map((u) => `<option value="${escapeHtml(u.slug)}">${escapeHtml((u.label || u.slug).toUpperCase())}</option>`).join('');
      if (existing && Array.from(useCaseSelect.options).some(o => o.value === existing)) {
        useCaseSelect.value = existing;
      }
    } catch (e) {
      console.error('use_cases fetch failed', e);
    }
  }

  async function loadSnapshot() {
    const q = useCaseQuery();
    try {
      const [reg, res, pol] = await Promise.all([
        fetch('/api/registry?' + q.slice(1)).then(r => r.json()),
        fetch('/api/results?n=200' + q).then(r => r.json()),
        fetch('/api/policy').then(r => r.json()),
      ]);
      renderPolicy(pol);
      renderRegistry(reg);
      logEvents = [];
      seenLogEvents.clear();
      logPanel.innerHTML = '';
      if (res && Array.isArray(res.rows)) {
        res.rows.forEach(addLogEvent);
      } else {
        renderLogPanel();
      }
    } catch (e) {
      console.error('snapshot fetch failed', e);
    }
  }

  function openStream() {
    if (currentEventSource) {
      try { currentEventSource.close(); } catch (_) {}
      currentEventSource = null;
    }
    setConn('reconnecting', 'CONNECTING…');
    const q = useCaseQuery();
    const es = new EventSource('/api/stream' + (q ? '?' + q.slice(1) : ''));
    currentEventSource = es;

    es.addEventListener('registry', (e) => {
      try { renderRegistry(JSON.parse(e.data)); }
      catch (err) { console.error('bad registry', err); }
    });

    es.addEventListener('result', (e) => {
      try { addLogEvent(JSON.parse(e.data)); }
      catch (err) { console.error('bad result', err); }
    });

    es.addEventListener('heartbeat', (e) => {
      try {
        const d = JSON.parse(e.data);
        lastHeartbeat = d.ts || Math.floor(Date.now() / 1000);
        heartbeatTs.textContent = fmtTs(lastHeartbeat);
      } catch (_) {}
    });

    es.onopen = () => {
      lastHeartbeat = Math.floor(Date.now() / 1000);
      setConn('connected', '● CONNECTED');
    };
    es.onerror = () => {
      setConn('reconnecting', '● RECONNECTING');
    };
  }

  async function refreshForSelection() {
    await loadSnapshot();
    openStream();
  }

  if (useCaseSelect) {
    useCaseSelect.addEventListener('change', () => { refreshForSelection(); });
  }

  setInterval(() => {
    if (!lastHeartbeat) return;
    const age = Math.floor(Date.now() / 1000) - lastHeartbeat;
    if (age > 30) setConn('disconnected', '● DISCONNECTED');
  }, 1000);

  detailClose.addEventListener('click', closeDetailModal);
  detailModal.addEventListener('click', (e) => {
    if (e.target === detailModal) closeDetailModal();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !detailModal.hidden) closeDetailModal();
  });

  populateUseCases().then(loadSnapshot).then(openStream);
})();
</script>

</body>
</html>
"""
