"""
QWEN plan layer — generates the next experiment queue.

plan(registry, runs, goal) → list[ExperimentPlan]

Calls a local QWEN model via Ollama (default: localhost:11434).
Returns a ranked list of experiments to run next.
Falls back to a rule-based ranker if QWEN is unavailable or
returns malformed JSON.

The planner is domain-agnostic. It reasons about patterns by
reading the registry and run history — it never imports domain code.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any

from memory.registry import PatternRegistry, RegistryEntry


OLLAMA_URL   = "http://localhost:11434/api/generate"
QWEN_MODEL   = "qwen2.5:latest"
MAX_TOKENS   = 1024
TEMPERATURE  = 0.3


@dataclass
class ExperimentPlan:
    """One planned experiment, as returned by the QWEN planner."""
    pattern_name: str          # which pattern to run
    rationale:    str          # why QWEN chose this pattern
    config:       dict         # hyperparameter suggestions
    priority:     int          # 1 = highest priority


def plan(
    registry:   PatternRegistry,
    run_history: list[dict],
    goal:        str = "maximise composite score",
    n_plans:     int = 3,
) -> list[ExperimentPlan]:
    """
    Generate the next N experiments to run.

    Tries QWEN first. Falls back to rule-based ranker on any failure.
    """
    try:
        return _qwen_plan(registry, run_history, goal, n_plans)
    except Exception:
        return _rule_based_plan(registry, n_plans)


def _build_prompt(
    registry:    PatternRegistry,
    run_history: list[dict],
    goal:        str,
    n_plans:     int,
) -> str:
    entries = registry.all()
    registry_summary = [
        {
            "pattern":   e.pattern,
            "tier":      e.tier,
            "confidence": e.confidence,
            "runs":      e.runs,
            "last_status": e.last_status,
            "promotion_candidate": e.promotion_candidate,
        }
        for e in entries
    ]

    recent_runs = run_history[-20:] if len(run_history) > 20 else run_history

    return f"""You are the planning layer of an autonomous experiment engine.

Goal: {goal}

Current pattern registry:
{json.dumps(registry_summary, indent=2)}

Recent experiment history (last {len(recent_runs)} runs):
{json.dumps(recent_runs, indent=2)}

Task: Suggest the next {n_plans} experiments to run.

Rules:
- Prioritise patterns with few runs and high potential
- Avoid repeating configs that already failed
- Suggest hyperparameter variations for promising patterns
- Consider patterns in 'scratch' tier that have never been tried
- Return ONLY valid JSON — no preamble, no markdown, no explanation

Return exactly this JSON structure:
{{
  "plans": [
    {{
      "pattern_name": "<name>",
      "rationale": "<one sentence why>",
      "config": {{}},
      "priority": 1
    }}
  ]
}}"""


def _qwen_plan(
    registry:    PatternRegistry,
    run_history: list[dict],
    goal:        str,
    n_plans:     int,
) -> list[ExperimentPlan]:
    prompt = _build_prompt(registry, run_history, goal, n_plans)

    payload = json.dumps({
        "model":  QWEN_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS,
        },
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read())

    raw = body.get("response", "")
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    parsed = json.loads(raw)

    return [
        ExperimentPlan(
            pattern_name = p["pattern_name"],
            rationale    = p.get("rationale", ""),
            config       = p.get("config", {}),
            priority     = p.get("priority", i + 1),
        )
        for i, p in enumerate(parsed.get("plans", []))
    ]


def _rule_based_plan(
    registry: PatternRegistry,
    n_plans:  int,
) -> list[ExperimentPlan]:
    """
    Fallback planner when QWEN is unavailable.

    Strategy:
      1. Patterns with zero runs (never tried) — highest priority
      2. Patterns below working_threshold with fewest runs — try more
      3. Patterns near stable_threshold — push over the line
    """
    entries = registry.all()
    plans: list[ExperimentPlan] = []

    for e in entries:
        if e.runs == 0:
            plans.append(ExperimentPlan(
                pattern_name = e.pattern,
                rationale    = "never been run — establish baseline",
                config       = {},
                priority     = len(plans) + 1,
            ))

    scratch = sorted(registry.by_tier("scratch"), key=lambda e: e.runs)
    for e in scratch:
        if len(plans) >= n_plans:
            break
        if not any(p.pattern_name == e.pattern for p in plans):
            plans.append(ExperimentPlan(
                pattern_name = e.pattern,
                rationale    = f"scratch tier, only {e.runs} runs — needs more data",
                config       = {},
                priority     = len(plans) + 1,
            ))

    working = sorted(
        registry.by_tier("working"),
        key=lambda e: e.confidence,
        reverse=True,
    )
    for e in working:
        if len(plans) >= n_plans:
            break
        plans.append(ExperimentPlan(
            pattern_name = e.pattern,
            rationale    = f"working tier, score {e.confidence:.3f} — push toward stable",
            config       = {},
            priority     = len(plans) + 1,
        ))

    return plans[:n_plans]
