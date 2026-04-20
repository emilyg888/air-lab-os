from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.promotion import promote_patterns


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply registry-driven pattern promotions.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply promotion in registry.json. Defaults to dry-run preview.",
    )
    parser.add_argument(
        "--reflect-filesystem",
        action="store_true",
        help="Also move pattern files so the filesystem matches the registry.",
    )
    args = parser.parse_args()

    actions = promote_patterns(
        dry_run=not args.apply,
        reflect_filesystem=args.reflect_filesystem,
    )
    if not actions:
        print("No promotion candidates found.")
        return

    for action in actions:
        if args.apply:
            if action.filesystem_reflected:
                prefix = "[MOVE]"
            elif action.applied:
                prefix = "[APPLY]"
            else:
                prefix = "[SKIP]"
            suffix = f" ({action.skipped_reason})" if action.skipped_reason else ""
        else:
            prefix = "[DRY]"
            suffix = ""
        print(
            f"{prefix} {action.source} -> {action.destination} "
            f"({action.pattern}: {action.current_status} -> {action.target_status}){suffix}"
        )


if __name__ == "__main__":
    main()
