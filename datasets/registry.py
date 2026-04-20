"""Dataset registry access for external dataset packages."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


def _candidate_registry_paths() -> list[Path]:
    repo_root = Path(__file__).resolve().parent.parent
    home = Path.home()
    return [
        repo_root.parent / "external-datasets" / "registry" / "datasets.json",
        repo_root.parent / "bb_datasets" / "external-datasets" / "registry" / "datasets.json",
        home / "LocalDocuments" / "Projects" / "external-datasets" / "registry" / "datasets.json",
        home / "LocalDocuments" / "Projects" / "bb_datasets" / "external-datasets" / "registry" / "datasets.json",
        home / "LocalDocuments" / "bb_datasets" / "external-datasets" / "registry" / "datasets.json",
    ]


def default_registry_path() -> Path:
    for path in _candidate_registry_paths():
        if path.exists():
            return path
    raise FileNotFoundError("No external dataset registry found")


@dataclass(frozen=True)
class DatasetDefinition:
    dataset_id: str
    domain: str
    path: Path
    version: str
    tier: str
    has_labels: bool
    metadata_path: Path
    schema_path: Path | None
    data_path: Path
    labels_path: Path | None


def load_registry(registry_path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    path = Path(registry_path) if registry_path is not None else default_registry_path()
    with path.open("r", encoding="utf-8") as handle:
        registry = json.load(handle)

    if not isinstance(registry, dict):
        raise ValueError("Dataset registry must be a JSON object keyed by dataset id")
    return registry


def get_dataset_definition(
    dataset_id: str,
    registry_path: str | Path | None = None,
) -> DatasetDefinition:
    path = Path(registry_path) if registry_path is not None else default_registry_path()
    registry = load_registry(path)
    payload = registry.get(dataset_id)
    if payload is None:
        raise KeyError(f"Dataset {dataset_id!r} not found in registry")
    return _build_definition(dataset_id, payload, path)


def _build_definition(
    dataset_id: str,
    payload: dict[str, Any],
    registry_path: Path,
) -> DatasetDefinition:
    registry_dir = registry_path.resolve().parent
    dataset_dir = (registry_dir / payload["path"]).resolve()

    metadata_path = dataset_dir / "metadata.json"
    schema_path = dataset_dir / "schema.json"
    labels_path = dataset_dir / "labels.csv"

    return DatasetDefinition(
        dataset_id=dataset_id,
        domain=payload["domain"],
        path=dataset_dir,
        version=str(payload["version"]),
        tier=payload["tier"],
        has_labels=bool(payload.get("has_labels", False)),
        metadata_path=metadata_path,
        schema_path=schema_path if schema_path.exists() else None,
        data_path=dataset_dir / "data.csv",
        labels_path=labels_path if labels_path.exists() else None,
    )
