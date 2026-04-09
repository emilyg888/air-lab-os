"""
air-lab-os — main entry point.

Usage:
    uv run python main.py status
    uv run python main.py run --pattern <name> --dataset <module>
    uv run python main.py arena --dataset <module>
    uv run python main.py loop --dataset <module>

The engine is domain-agnostic. Use-case plugins are imported
dynamically via --dataset. Patterns are discovered from the
patterns/ directory tree.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from memory.registry import PatternRegistry, RUNS_PATH, REGISTRY_PATH


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
    registry = PatternRegistry.load()
    entries  = registry.all()
    if not entries:
        print("Registry is empty — no experiments run yet.")
        return
    print(f"\n{'Pattern':<30} {'Tier':<10} {'Score':>7} {'Runs':>5} {'Status':<10}")
    print("-" * 68)
    for e in entries:
        cand = " [promote?]" if e.promotion_candidate else ""
        print(
            f"{e.pattern:<30} {e.tier:<10} "
            f"{e.confidence:>7.4f} {e.runs:>5} "
            f"{e.last_status:<10}{cand}"
        )
    candidates = registry.promotion_candidates()
    if candidates:
        print(f"\n  Promotion candidates: {[e.pattern for e in candidates]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="air-lab-os — domain-agnostic self-improving engine"
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("status", help="Print registry state")

    run_p = subparsers.add_parser("run", help="Run one experiment")
    run_p.add_argument("--pattern",     required=True, help="Pattern name")
    run_p.add_argument("--dataset",     required=True, help="Dataset plugin module")
    run_p.add_argument("--description", default="",    help="Run description")

    arena_p = subparsers.add_parser("arena", help="Compare all patterns")
    arena_p.add_argument("--dataset", required=True, help="Dataset plugin module")

    loop_p = subparsers.add_parser("loop", help="Run autonomous experiment loop")
    loop_p.add_argument("--dataset", required=True, help="Dataset plugin module")
    loop_p.add_argument("--goal",    default="maximise composite score")
    loop_p.add_argument("--rounds",  type=int, default=0,
                         help="Max rounds (0 = run until interrupted)")

    args = parser.parse_args()

    if args.command == "status" or args.command is None:
        cmd_status()
        return

    dataset_module = importlib.import_module(args.dataset)
    handle = dataset_module.get_handle()

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
