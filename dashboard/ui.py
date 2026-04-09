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
  header .use-case .value {
    color: var(--green);
    font-weight: 700;
  }

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
  .panel-body { flex: 1; overflow-y: auto; padding: 10px 12px; }
  .panel-body::-webkit-scrollbar { width: 8px; }
  .panel-body::-webkit-scrollbar-thumb { background: var(--border); }

  /* Promotion banner */
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

  /* Registry cards */
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

  /* Log rows */
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
  .log-row .marker.keep { color: var(--green); }
  .log-row .marker.discard { color: #8a6a1f; opacity: 0.7; }
  .log-row .marker.crash { color: var(--red); }
  .log-row-detail {
    display: none;
    padding: 8px 12px 10px 80px;
    color: var(--text-dim);
    font-size: 11px;
    border-bottom: 1px solid rgba(31, 34, 51, 0.5);
    background: #0c0e17;
  }
  .log-row-detail.open { display: block; }
  .log-row-detail pre {
    margin: 0;
    white-space: pre-wrap;
    color: var(--text-dim);
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
</style>
</head>
<body>

<header>
  <span class="brand">AIR LAB OS</span>
  <span class="dot"></span>
  <span class="live">LIVE</span>
  <span class="use-case" id="use-case-box">
    <span class="label">USE CASE</span>
    <span class="value" id="use-case-value">—</span>
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
    <div class="panel-title">EXPERIMENT LOG</div>
    <div class="panel-body" id="log-panel">
      <div class="empty">awaiting events…</div>
    </div>
    <div class="connection disconnected" id="conn">
      <span class="dot"></span>
      <span id="conn-label">CONNECTING…</span>
    </div>
  </section>
</main>

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
  const useCaseValue  = document.getElementById('use-case-value');
  const heartbeatTs   = document.getElementById('heartbeat-ts');
  const conn          = document.getElementById('conn');
  const connLabel     = document.getElementById('conn-label');
  const banner        = document.getElementById('promotion-banner');

  let lastHeartbeat = 0;
  let thresholds = { working: WORKING, stable: STABLE };

  const fmtTs = (s) => {
    if (!s) return '—';
    if (typeof s === 'number') s = new Date(s * 1000).toISOString();
    return String(s).slice(11, 19) || String(s).slice(0, 19);
  };

  const escapeHtml = (str) =>
    String(str).replace(/[&<>"']/g, (c) =>
      ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  function renderRegistry(snapshot) {
    if (snapshot.thresholds) thresholds = snapshot.thresholds;
    const patterns = snapshot.patterns || [];
    headerTs.textContent = new Date().toISOString().slice(0, 19) + 'Z';
    if (snapshot.use_case) {
      const uc = snapshot.use_case;
      useCaseValue.textContent = uc.current
        ? uc.current.toUpperCase()
        : 'UNKNOWN';
      useCaseValue.title = uc.all && uc.all.length
        ? 'all: ' + uc.all.join(', ')
        : '';
    }
    if (!patterns.length) {
      registryPanel.innerHTML = '<div class="empty">no patterns yet</div>';
      banner.classList.remove('active');
      return;
    }
    const working = thresholds.working || WORKING;
    const stable  = thresholds.stable  || STABLE;
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
    registryPanel.innerHTML = html;

    const hasCand = (snapshot.promotion_candidates || []).length > 0;
    banner.classList.toggle('active', hasCand);
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

  function prependLogRow(ev) {
    if (logPanel.querySelector('.empty')) logPanel.innerHTML = '';
    const marker = statusToMarker(ev.status || 'keep');
    const ts = fmtTs(ev.ts || ev.last_updated);
    const row = document.createElement('div');
    row.className = 'log-row';
    row.innerHTML = `
      <span class="ts">${escapeHtml(ts)}</span>
      <span class="pat">${escapeHtml(ev.pattern || '?')}</span>
      <span class="score">${Number(ev.score || 0).toFixed(4)}</span>
      <span class="marker ${marker.cls}">${marker.char}</span>
    `;
    const detail = document.createElement('div');
    detail.className = 'log-row-detail';
    detail.innerHTML = '<pre>' + escapeHtml(JSON.stringify(ev, null, 2)) + '</pre>';
    row.addEventListener('click', () => detail.classList.toggle('open'));
    logPanel.insertBefore(detail, logPanel.firstChild);
    logPanel.insertBefore(row, logPanel.firstChild);

    // Trim to MAX_LOG_ROWS pairs (row + detail)
    while (logPanel.children.length > MAX_LOG_ROWS * 2) {
      logPanel.removeChild(logPanel.lastChild);
    }
  }

  function setConn(state, label) {
    conn.classList.remove('connected', 'reconnecting', 'disconnected');
    conn.classList.add(state);
    connLabel.textContent = label;
  }

  async function initialFetch() {
    try {
      const [reg, res, pol] = await Promise.all([
        fetch('/api/registry').then(r => r.json()),
        fetch('/api/results?n=20').then(r => r.json()),
        fetch('/api/policy').then(r => r.json()),
      ]);
      renderPolicy(pol);
      renderRegistry(reg);
      if (res && Array.isArray(res.rows)) {
        // Render oldest first so newest ends up on top after prepending.
        const ordered = res.rows.slice().reverse();
        ordered.forEach(prependLogRow);
      }
    } catch (e) {
      console.error('initial fetch failed', e);
    }
  }

  function openStream() {
    setConn('reconnecting', 'CONNECTING…');
    const es = new EventSource('/api/stream');

    es.addEventListener('registry', (e) => {
      try { renderRegistry(JSON.parse(e.data)); }
      catch (err) { console.error('bad registry', err); }
    });

    es.addEventListener('result', (e) => {
      try { prependLogRow(JSON.parse(e.data)); }
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

  setInterval(() => {
    if (!lastHeartbeat) return;
    const age = Math.floor(Date.now() / 1000) - lastHeartbeat;
    if (age > 30) setConn('disconnected', '● DISCONNECTED');
  }, 1000);

  initialFetch().then(openStream);
})();
</script>

</body>
</html>
"""
