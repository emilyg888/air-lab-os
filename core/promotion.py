"""Controlled pattern promotion based on registry proposals."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from core.registry import REGISTRY_PATH, apply_promotion, load_registry

BASE_DIR = Path("patterns")
PROMOTION_PATHS = {
    "silver": ("scratch", "working"),
    "gold": ("working", "stable"),
}


@dataclass
class PromotionAction:
    pattern: str
    current_status: str
    target_status: str
    source: Path
    destination: Path
    applied: bool = False
    filesystem_reflected: bool = False
    skipped_reason: str = ""


def plan_promotions(
    registry_path: Path = REGISTRY_PATH,
    base_dir: Path = BASE_DIR,
) -> list[PromotionAction]:
    registry = load_registry(registry_path)
    actions: list[PromotionAction] = []

    for pattern, entry in sorted(registry.items()):
        target_status = entry.get("promotion_candidate")
        if target_status not in PROMOTION_PATHS:
            continue

        src_dir, dst_dir = PROMOTION_PATHS[target_status]
        actions.append(
            PromotionAction(
                pattern=pattern,
                current_status=str(entry.get("status", "bronze")),
                target_status=str(target_status),
                source=base_dir / src_dir / f"{pattern}.py",
                destination=base_dir / dst_dir / f"{pattern}.py",
            )
        )

    return actions


def promote_patterns(
    dry_run: bool = True,
    reflect_filesystem: bool = False,
    registry_path: Path = REGISTRY_PATH,
    base_dir: Path = BASE_DIR,
) -> list[PromotionAction]:
    actions = plan_promotions(registry_path=registry_path, base_dir=base_dir)

    if dry_run:
        return actions

    for action in actions:
        if reflect_filesystem:
            if not action.source.exists():
                action.skipped_reason = "source file is missing"
                continue
            if action.destination.exists():
                action.skipped_reason = "destination file already exists"
                continue

            action.destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(action.source), str(action.destination))
            action.filesystem_reflected = True

        apply_promotion(
            pattern_name=action.pattern,
            target_status=action.target_status,
            path=registry_path,
        )
        action.applied = True

    return actions
