"""Dataset loading for registered dataset ids and module-path handles."""

from __future__ import annotations

import importlib
from dataclasses import replace

from datasets.base import DatasetHandle, DatasetMeta
from datasets.registry import get_dataset_definition


class AliasedDatasetHandle(DatasetHandle):
    def __init__(self, alias: str, delegate: DatasetHandle):
        self._alias = alias
        self._delegate = delegate

    @property
    def meta(self) -> DatasetMeta:
        return replace(self._delegate.meta, name=self._alias)

    def eval_df(self):
        return self._delegate.eval_df()

    def train_df(self):
        return self._delegate.train_df()

    def labels(self) -> list[bool]:
        return self._delegate.labels()


def _load_registered_dataset(dataset_name: str) -> DatasetHandle:
    definition = get_dataset_definition(dataset_name)

    if definition.domain == "fraud" or definition.dataset_id == "edge_cases_v1":
        module = importlib.import_module("use_cases.fraud.registry_handle")
        return module.get_handle(definition)

    raise ValueError(
        f"Dataset {dataset_name!r} is registered for domain {definition.domain!r}, "
        "but no DatasetHandle adapter is available for that domain"
    )


def load_dataset(dataset_name: str) -> DatasetHandle:
    try:
        return _load_registered_dataset(dataset_name)
    except KeyError:
        pass
    except FileNotFoundError:
        pass

    module = importlib.import_module(dataset_name)
    handle = module.get_handle()

    if handle.meta.name == dataset_name:
        return handle
    return AliasedDatasetHandle(dataset_name, handle)
