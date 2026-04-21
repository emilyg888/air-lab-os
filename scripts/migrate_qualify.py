"""One-shot migration: qualify pattern identifiers by use case.

Reads memory/runs.json, infers `use_case` from `dataset_id`, writes back
with `use_case` populated on every record (idempotent), then rebuilds
registry.json from the migrated runs via PatternRegistry.load().

Usage:
    uv run python -m scripts.migrate_qualify
    uv run python -m scripts.migrate_qualify --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_PATH = REPO_ROOT / "memory" / "runs.json"
REGISTRY_PATH = REPO_ROOT / "registry.json"

_DATASET_USE_CASE = re.compile(r"^use_cases\.([^.]+)\.")


def use_case_from_dataset_id(dataset_id: str) -> str:
    if not dataset_id:
        return "_unknown"
    m = _DATASET_USE_CASE.match(dataset_id)
    return m.group(1) if m else "_unknown"


def migrate_runs(runs: list[dict]) -> tuple[list[dict], dict[str, int]]:
    stats = {"total": len(runs), "tagged": 0, "already_tagged": 0, "unknown": 0}
    out: list[dict] = []
    for run in runs:
        new = dict(run)
        existing = new.get("use_case")
        if existing:
            stats["already_tagged"] += 1
        else:
            slug = use_case_from_dataset_id(new.get("dataset_id", ""))
            new["use_case"] = slug
            if slug == "_unknown":
                stats["unknown"] += 1
            else:
                stats["tagged"] += 1
        out.append(new)
    return out, stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runs", default=str(RUNS_PATH))
    parser.add_argument("--registry", default=str(REGISTRY_PATH))
    args = parser.parse_args()

    runs_path = Path(args.runs)
    registry_path = Path(args.registry)

    if not runs_path.exists():
        print(f"no runs.json at {runs_path} — nothing to migrate")
        return 0

    runs = json.loads(runs_path.read_text())
    if not isinstance(runs, list):
        print(f"runs.json is not a list: {type(runs).__name__}", file=sys.stderr)
        return 1

    migrated, stats = migrate_runs(runs)
    print(
        f"runs: total={stats['total']} tagged={stats['tagged']} "
        f"already_tagged={stats['already_tagged']} unknown={stats['unknown']}"
    )

    if args.dry_run:
        print("dry run — no files written")
        return 0

    if stats["already_tagged"] == stats["total"] and stats["total"] > 0:
        print("all records already tagged — nothing to do (skipping file write)")
        return 0

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    runs_backup = runs_path.with_name(f"{runs_path.name}.bak-{ts}")
    shutil.copy2(runs_path, runs_backup)
    print(f"backed up {runs_path} -> {runs_backup}")
    if registry_path.exists():
        reg_backup = registry_path.with_name(f"{registry_path.name}.bak-{ts}")
        shutil.copy2(registry_path, reg_backup)
        print(f"backed up {registry_path} -> {reg_backup}")

    runs_path.write_text(json.dumps(migrated, indent=2))
    print(f"wrote {runs_path}")

    from core.registry import PatternRegistry

    reg = PatternRegistry.load(runs_path=runs_path, registry_path=registry_path)
    reg.save(registry_path)
    print(f"rebuilt {registry_path} — {len(reg.all())} qualified patterns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
