"""Generate a static dashboard page for fraud feature experiment results."""

from __future__ import annotations

import argparse
from datetime import datetime
from html import escape
from pathlib import Path

from core.dataset_loader import load_dataset
from use_cases.fraud.feature_lab.run_feature_experiment import (
    FeatureExperimentResult,
    run_feature_experiments,
)


DEFAULT_DATASET = "use_cases.fraud.handle"
DEFAULT_OUTPUT = Path("dashboard/feature_results.html")
_DELTA_EPSILON = 1e-9


def _bucket(delta_f1: float) -> str:
    if delta_f1 > _DELTA_EPSILON:
        return "improved"
    if delta_f1 < -_DELTA_EPSILON:
        return "regressed"
    return "flat"


def _bucket_label(delta_f1: float) -> str:
    bucket = _bucket(delta_f1)
    if bucket == "improved":
        return "Improved"
    if bucket == "regressed":
        return "Regressed"
    return "Flat"


def _summary_counts(results: list[FeatureExperimentResult]) -> dict[str, int]:
    counts = {"improved": 0, "flat": 0, "regressed": 0}
    for result in results:
        counts[_bucket(result.delta_f1)] += 1
    return counts


def render_feature_dashboard(
    *,
    dataset_name: str,
    baseline_f1: float,
    results: list[FeatureExperimentResult],
    generated_at: str,
) -> str:
    ordered = sorted(results, key=lambda result: result.delta_f1, reverse=True)
    counts = _summary_counts(ordered)
    best_delta = ordered[0].delta_f1 if ordered else 0.0
    worst_delta = ordered[-1].delta_f1 if ordered else 0.0

    cards = []
    for result in ordered:
        bucket = _bucket(result.delta_f1)
        cards.append(
            f"""
            <article class="feature-card {bucket}">
              <div class="feature-card__top">
                <h3>{escape(result.feature_name)}</h3>
                <span class="pill {bucket}">{_bucket_label(result.delta_f1)}</span>
              </div>
              <div class="feature-card__delta">{result.delta_f1:+.4f}</div>
              <p class="feature-card__sub">New F1 {result.experiment_f1:.4f} from baseline {baseline_f1:.4f}</p>
              <p class="feature-card__meta">Model feature count: {result.feature_count}</p>
            </article>
            """.strip()
        )

    rows = []
    for idx, result in enumerate(ordered, start=1):
        bucket = _bucket(result.delta_f1)
        rows.append(
            f"""
            <tr>
              <td>{idx}</td>
              <td><code>{escape(result.feature_name)}</code></td>
              <td>{result.experiment_f1:.4f}</td>
              <td class="{bucket}">{result.delta_f1:+.4f}</td>
              <td>{_bucket_label(result.delta_f1)}</td>
            </tr>
            """.strip()
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Feature Experiment Dashboard</title>
  <style>
    :root {{
      --bg: #f4efe7;
      --bg-panel: rgba(255, 252, 247, 0.9);
      --bg-strong: #1f3a5f;
      --ink: #1f1c18;
      --muted: #6f665b;
      --line: rgba(31, 28, 24, 0.12);
      --improved: #0f766e;
      --flat: #9a6700;
      --regressed: #b42318;
      --accent: #d98f2b;
      --shadow: 0 16px 48px rgba(58, 42, 25, 0.12);
      --font-display: "Avenir Next", "Segoe UI", sans-serif;
      --font-body: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
      --font-mono: "SF Mono", "Fira Code", monospace;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at top left, rgba(217, 143, 43, 0.24), transparent 32%),
        linear-gradient(160deg, #f8f3ea 0%, #ede3d5 52%, #f4efe7 100%);
      color: var(--ink);
      font-family: var(--font-body);
      padding: 32px 20px 48px;
    }}
    .wrap {{
      max-width: 1120px;
      margin: 0 auto;
    }}
    .hero {{
      background: var(--bg-panel);
      backdrop-filter: blur(14px);
      border: 1px solid rgba(255, 255, 255, 0.4);
      border-radius: 24px;
      box-shadow: var(--shadow);
      padding: 28px;
      margin-bottom: 24px;
    }}
    .eyebrow {{
      font: 600 12px var(--font-display);
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 12px;
    }}
    h1 {{
      margin: 0 0 12px;
      font: 700 clamp(34px, 6vw, 62px)/0.95 var(--font-display);
      letter-spacing: -0.04em;
    }}
    .hero p {{
      margin: 0;
      max-width: 760px;
      font-size: 18px;
      line-height: 1.55;
      color: var(--muted);
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 14px;
      margin: 22px 0 0;
    }}
    .stat {{
      background: rgba(255, 255, 255, 0.66);
      border: 1px solid rgba(255, 255, 255, 0.58);
      border-radius: 18px;
      padding: 16px;
    }}
    .stat__label {{
      font: 600 11px var(--font-display);
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .stat__value {{
      font: 700 30px/1 var(--font-display);
      letter-spacing: -0.04em;
    }}
    .section-title {{
      font: 700 20px var(--font-display);
      margin: 0 0 14px;
    }}
    .card-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 24px;
    }}
    .feature-card {{
      background: rgba(255, 252, 247, 0.94);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 18px;
      box-shadow: 0 12px 28px rgba(82, 57, 29, 0.08);
    }}
    .feature-card__top {{
      display: flex;
      justify-content: space-between;
      align-items: start;
      gap: 12px;
      margin-bottom: 16px;
    }}
    .feature-card h3 {{
      margin: 0;
      font: 700 20px/1.1 var(--font-display);
      letter-spacing: -0.03em;
    }}
    .feature-card__delta {{
      font: 700 40px/1 var(--font-display);
      letter-spacing: -0.06em;
      margin-bottom: 10px;
    }}
    .feature-card__sub,
    .feature-card__meta {{
      margin: 0;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.45;
    }}
    .feature-card__meta {{
      margin-top: 6px;
      font-size: 13px;
    }}
    .pill {{
      border-radius: 999px;
      padding: 7px 10px;
      font: 700 11px var(--font-display);
      letter-spacing: 0.08em;
      text-transform: uppercase;
      white-space: nowrap;
    }}
    .improved {{ color: var(--improved); }}
    .flat {{ color: var(--flat); }}
    .regressed {{ color: var(--regressed); }}
    .pill.improved {{ background: rgba(15, 118, 110, 0.12); }}
    .pill.flat {{ background: rgba(154, 103, 0, 0.14); }}
    .pill.regressed {{ background: rgba(180, 35, 24, 0.12); }}
    .table-wrap {{
      background: rgba(255, 252, 247, 0.94);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 18px;
      overflow-x: auto;
      box-shadow: 0 12px 28px rgba(82, 57, 29, 0.08);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 15px;
    }}
    th, td {{
      text-align: left;
      padding: 14px 10px;
      border-bottom: 1px solid var(--line);
    }}
    th {{
      font: 700 12px var(--font-display);
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
    }}
    td code {{
      font-family: var(--font-mono);
      font-size: 13px;
    }}
    .footer {{
      margin-top: 14px;
      color: var(--muted);
      font-size: 13px;
    }}
    @media (max-width: 920px) {{
      .stats,
      .card-grid {{
        grid-template-columns: 1fr 1fr;
      }}
    }}
    @media (max-width: 640px) {{
      body {{ padding: 18px 14px 32px; }}
      .hero {{ padding: 22px; }}
      .stats,
      .card-grid {{
        grid-template-columns: 1fr;
      }}
      h1 {{ font-size: 36px; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="eyebrow">Feature Experiment Dashboard</div>
      <h1>Seven candidate signals against <code>{escape(dataset_name)}</code></h1>
      <p>The current fraud feature pass retrains <code>ml_logistic</code> with one additional feature at a time, then measures the resulting F1 delta against the existing baseline.</p>
      <div class="stats">
        <div class="stat">
          <div class="stat__label">Baseline F1</div>
          <div class="stat__value">{baseline_f1:.4f}</div>
        </div>
        <div class="stat">
          <div class="stat__label">Candidates</div>
          <div class="stat__value">{len(ordered)}</div>
        </div>
        <div class="stat">
          <div class="stat__label">Improved</div>
          <div class="stat__value improved">{counts["improved"]}</div>
        </div>
        <div class="stat">
          <div class="stat__label">Flat</div>
          <div class="stat__value flat">{counts["flat"]}</div>
        </div>
        <div class="stat">
          <div class="stat__label">Regressed</div>
          <div class="stat__value regressed">{counts["regressed"]}</div>
        </div>
      </div>
      <div class="stats">
        <div class="stat">
          <div class="stat__label">Best Delta</div>
          <div class="stat__value { _bucket(best_delta) }">{best_delta:+.4f}</div>
        </div>
        <div class="stat">
          <div class="stat__label">Worst Delta</div>
          <div class="stat__value { _bucket(worst_delta) }">{worst_delta:+.4f}</div>
        </div>
        <div class="stat" style="grid-column: span 3;">
          <div class="stat__label">Generated</div>
          <div class="stat__value" style="font-size:18px; letter-spacing:0;">{escape(generated_at)}</div>
        </div>
      </div>
    </section>

    <section>
      <h2 class="section-title">Feature Cards</h2>
      <div class="card-grid">
        {' '.join(cards)}
      </div>
    </section>

    <section class="table-wrap">
      <h2 class="section-title">Ranked Results</h2>
      <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Feature</th>
            <th>New F1</th>
            <th>Delta F1</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {' '.join(rows)}
        </tbody>
      </table>
      <div class="footer">Current pass summary: no feature improved baseline F1; the best observed outcome was neutral.</div>
    </section>
  </div>
</body>
</html>
"""


def write_feature_dashboard(
    output_path: Path = DEFAULT_OUTPUT,
    dataset_name: str = DEFAULT_DATASET,
) -> Path:
    handle = load_dataset(dataset_name)
    results = run_feature_experiments(handle)
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    html = render_feature_dashboard(
        dataset_name=handle.meta.name,
        baseline_f1=results[0].baseline_f1 if results else 0.0,
        results=results,
        generated_at=generated_at,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate feature experiment dashboard HTML")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset id or DatasetHandle module path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output HTML path")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output = write_feature_dashboard(
        output_path=Path(args.output),
        dataset_name=args.dataset,
    )
    print(output)


if __name__ == "__main__":
    main()
