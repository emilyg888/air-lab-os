"""
air-lab-os — main entry point.

Usage:
    uv run python main.py status
    uv run python main.py run --pattern <name> --dataset <module>
    uv run python main.py arena --dataset <module>
    uv run python main.py loop --dataset <module>
    uv run python scripts/promote.py [--apply] [--reflect-filesystem]

The engine is domain-agnostic. Use-case plugins are imported
dynamically via --dataset. Patterns are discovered from the
patterns/ directory tree.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from core.dataset_loader import load_dataset
from core.registry import PatternRegistry, RUNS_PATH, REGISTRY_PATH, load_registry


def _load_runs() -> list[dict]:
    if RUNS_PATH.exists():
        try:
            return json.loads(RUNS_PATH.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    return []


def _discover_patterns() -> dict[str, Path]:
    """Find all pattern .py files across tier directories."""
    found = {}
    for tier_dir in ["patterns/scratch", "patterns/working", "patterns/stable"]:
        for p in Path(tier_dir).glob("*.py"):
            if p.name.startswith("_"):
                continue
            found[p.stem] = p
    return found


def cmd_status() -> None:
    """Print current registry state."""
    registry = load_registry()
    if not registry:
        print("Registry is empty — no experiments run yet.")
        return

    entries = sorted(
        registry.items(),
        key=lambda item: item[1].get("confidence", 0.0),
        reverse=True,
    )

    print(
        f"\n{'Pattern':<30} {'Status':<10} {'Candidate':<10} "
        f"{'Avg':>7} {'Last':>7} {'Runs':>5} {'Stable':<6}"
    )
    print("-" * 88)
    for pattern, entry in entries:
        print(
            f"{pattern:<30} {entry.get('status', 'bronze'):<10} "
            f"{(entry.get('promotion_candidate') or '-'):<10} "
            f"{entry.get('avg_score', 0.0):>7.4f} {entry.get('last_score', 0.0):>7.4f} "
            f"{entry.get('runs', 0):>5} {str(entry.get('is_stable', False)):<6}"
        )
    candidates = [
        pattern for pattern, entry in entries
        if entry.get("promotion_candidate")
    ]
    if candidates:
        print(f"\n  Promotion candidates: {candidates}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="air-lab-os — domain-agnostic self-improving engine"
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("status", help="Print registry state")

    run_p = subparsers.add_parser("run", help="Run one experiment")
    run_p.add_argument("--pattern",     required=True, help="Pattern name")
    run_p.add_argument("--dataset",     required=True, help="Dataset id or DatasetHandle module")
    run_p.add_argument("--description", default="",    help="Run description")

    arena_p = subparsers.add_parser("arena", help="Compare all patterns")
    arena_p.add_argument("--dataset", required=True, help="Dataset id or DatasetHandle module")

    loop_p = subparsers.add_parser("loop", help="Run autonomous experiment loop")
    loop_p.add_argument("--dataset", required=True, help="Dataset id or DatasetHandle module")
    loop_p.add_argument("--goal",    default="maximise composite score")
    loop_p.add_argument("--rounds",  type=int, default=0,
                         help="Max rounds (0 = run until interrupted)")

    args = parser.parse_args()

    if args.command == "status" or args.command is None:
        cmd_status()
        return

    handle = load_dataset(args.dataset)

    if args.command == "run":
        from lab.playground import run_experiment
        patterns = _discover_patterns()
        if args.pattern not in patterns:
            print(f"Pattern '{args.pattern}' not found. Available: {list(patterns)}")
            return
        module = importlib.import_module(
            patterns[args.pattern].as_posix()
            .replace("/", ".").removesuffix(".py")
        )
        pattern = module.get_pattern()
        result  = run_experiment(pattern, handle, description=args.description)
        print(f"\n--- Run result ---")
        print(f"pattern:  {result.pattern}")
        print(f"status:   {result.status}")
        print(f"tier:     {result.tier}")
        print(f"score:    {result.score:.4f}")
        if result.metrics:
            m = result.metrics
            print(f"primary:  {m.primary_metric_value:.4f}")
            print(f"expl:     {m.explainability_score:.4f}")
            print(f"latency:  {m.latency_score:.4f}")
        print(f"commit:   {result.commit}")

    elif args.command == "arena":
        from lab.arena import compare_patterns
        patterns = _discover_patterns()
        pattern_instances = []
        for name, path in patterns.items():
            mod = importlib.import_module(
                path.as_posix().replace("/", ".").removesuffix(".py")
            )
            pattern_instances.append(mod.get_pattern())
        arena = compare_patterns(pattern_instances, handle)
        print(f"\n--- Arena results ({handle.meta.name}) ---")
        for i, r in enumerate(arena.rankings, 1):
            crash = " [CRASH]" if r.status == "crash" else ""
            print(f"  {i}. {r.pattern:<28} score={r.score:.4f}{crash}")
        if arena.winner:
            print(f"\n  Winner: {arena.winner.pattern}")

    elif args.command == "loop":
        from lab.playground import run_experiment
        from runtime.llm import plan
        round_n = 0
        print(f"Starting autonomous loop. Goal: {args.goal}")
        print("Press Ctrl+C to stop.\n")
        try:
            while args.rounds == 0 or round_n < args.rounds:
                round_n += 1
                registry = PatternRegistry.load()
                runs     = _load_runs()
                plans    = plan(registry, runs, goal=args.goal)
                if not plans:
                    print("Planner returned no experiments. Stopping.")
                    break
                for p in plans:
                    patterns = _discover_patterns()
                    if p.pattern_name not in patterns:
                        print(f"  [skip] {p.pattern_name} — not found in patterns/")
                        continue
                    mod = importlib.import_module(
                        patterns[p.pattern_name].as_posix()
                        .replace("/", ".").removesuffix(".py")
                    )
                    pattern = mod.get_pattern()
                    print(f"  Round {round_n}: {p.pattern_name} — {p.rationale}")
                    result = run_experiment(pattern, handle, description=p.rationale)
                    print(
                        f"    score={result.score:.4f} "
                        f"status={result.status} "
                        f"tier={result.tier}"
                    )
        except KeyboardInterrupt:
            print("\nLoop interrupted by user.")
        cmd_status()


if __name__ == "__main__":
    main()
