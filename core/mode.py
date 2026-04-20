"""Shared execution modes."""

from __future__ import annotations

from enum import StrEnum


class Mode(StrEnum):
    EXPLORATION = "exploration"
    EXECUTION = "execution"


def normalize_mode(mode: Mode | str | None) -> Mode:
    if isinstance(mode, Mode):
        return mode
    if mode is None:
        return Mode.EXECUTION
    return Mode(mode)
