"""DatasetHandle adapters for registry-backed fraud datasets."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from datasets.base import DatasetHandle, DatasetMeta
from datasets.registry import DatasetDefinition
from use_cases.fraud.handle import build_features

_SPLIT_RATIO = 0.8
_FULL_FRAME_MIN_ROWS = 10


def _load_registry_payload(definition: DatasetDefinition) -> tuple[pd.DataFrame, dict, pd.DataFrame | None]:
    data = pd.read_csv(definition.data_path)
    metadata = json.loads(definition.metadata_path.read_text())
    labels = pd.read_csv(definition.labels_path) if definition.labels_path is not None else None
    return data, metadata, labels


def _normalize_transaction_frame(
    data: pd.DataFrame,
    metadata: dict,
    labels: pd.DataFrame | None,
) -> pd.DataFrame:
    frame = data.copy()
    primary_keys = metadata.get("primary_keys") or []
    label_column = metadata.get("label_column")

    if labels is not None and label_column and primary_keys:
        frame = frame.merge(labels, on=primary_keys, how="left")

    rename_map = {}
    if "transaction_id" in frame.columns:
        rename_map["transaction_id"] = "txn_id"
    if label_column and label_column in frame.columns:
        rename_map[label_column] = "fraud_flag"
    frame = frame.rename(columns=rename_map)

    if "txn_type" not in frame.columns:
        frame["txn_type"] = "DEBIT"
    if "account_id" not in frame.columns:
        frame["account_id"] = [f"acct_{idx:05d}" for idx in range(len(frame))]
    if "fraud_flag" not in frame.columns:
        frame["fraud_flag"] = False

    frame["fraud_flag"] = frame["fraud_flag"].fillna(False).astype(bool)
    return build_features(frame)


def _edge_case_transactions(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    base_timestamp = pd.Timestamp("2024-05-01T00:00:00")

    for idx, case in data.reset_index(drop=True).iterrows():
        case_id = str(case["case_id"])
        severity = case.get("severity", "low")
        signal = case.get("expected_signal", "none")
        account_id = f"edge_acct_{idx + 1:03d}"

        if signal == "velocity":
            for offset in range(5):
                rows.append(
                    {
                        "txn_id": f"{case_id}_{offset + 1}",
                        "account_id": account_id,
                        "amount": 250.0,
                        "timestamp": base_timestamp.isoformat(),
                        "txn_type": "DEBIT",
                        "fraud_flag": True,
                        "case_id": case_id,
                        "expected_signal": signal,
                        "severity": severity,
                    }
                )
        elif signal == "spike":
            rows.append(
                {
                    "txn_id": f"{case_id}_1",
                    "account_id": account_id,
                    "amount": 12000.0,
                    "timestamp": (base_timestamp + pd.Timedelta(minutes=idx)).isoformat(),
                    "txn_type": "DEBIT",
                    "fraud_flag": True,
                    "case_id": case_id,
                    "expected_signal": signal,
                    "severity": severity,
                }
            )
        else:
            rows.append(
                {
                    "txn_id": f"{case_id}_1",
                    "account_id": account_id,
                    "amount": 35.0,
                    "timestamp": (base_timestamp + pd.Timedelta(minutes=idx)).isoformat(),
                    "txn_type": "DEBIT",
                    "fraud_flag": False,
                    "case_id": case_id,
                    "expected_signal": signal,
                    "severity": severity,
                }
            )

    return build_features(pd.DataFrame(rows))


def _sort_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "txn_id" in frame.columns:
        return frame.sort_values("txn_id").reset_index(drop=True)
    return frame.reset_index(drop=True)


def _should_use_full_frame(frame: pd.DataFrame, label_column: str) -> bool:
    if len(frame) < _FULL_FRAME_MIN_ROWS:
        return True

    if label_column not in frame.columns:
        return True

    split_idx = int(len(frame) * _SPLIT_RATIO)
    split_idx = min(max(split_idx, 1), len(frame) - 1)
    train = frame.iloc[:split_idx]
    eval_ = frame.iloc[split_idx:]

    return (
        train[label_column].nunique(dropna=False) < 2
        or eval_[label_column].nunique(dropna=False) < 2
    )


def _split_frame(frame: pd.DataFrame, label_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = _sort_frame(frame)

    if len(ordered) <= 1 or _should_use_full_frame(ordered, label_column):
        shared = ordered.reset_index(drop=True)
        return shared.copy(), shared.copy()

    split_idx = int(len(ordered) * _SPLIT_RATIO)
    split_idx = min(max(split_idx, 1), len(ordered) - 1)
    train = ordered.iloc[:split_idx].reset_index(drop=True)
    eval_ = ordered.iloc[split_idx:].reset_index(drop=True)
    return train, eval_


class ExternalFraudDatasetHandle(DatasetHandle):
    def __init__(self, definition: DatasetDefinition):
        self._definition = definition
        self._data, self._metadata, self._labels = _load_registry_payload(definition)

        if definition.dataset_id == "edge_cases_v1":
            frame = _edge_case_transactions(self._data)
            self._label_column = "fraud_flag"
            self._primary_metric = "f1_score"
            self._domain = "fraud"
        else:
            frame = _normalize_transaction_frame(self._data, self._metadata, self._labels)
            self._label_column = "fraud_flag"
            self._primary_metric = self._metadata.get("evaluation_metric", "f1_score")
            self._domain = self._metadata.get("domain", definition.domain)

        self._full = _sort_frame(frame)
        self._train, self._eval = _split_frame(frame, self._label_column)

    @property
    def meta(self) -> DatasetMeta:
        return DatasetMeta(
            name=self._definition.dataset_id,
            domain=self._domain,
            tier=self._definition.tier,
            version=self._definition.version,
            label_column=self._label_column,
            primary_metric=self._primary_metric,
            row_count=len(self._full),
            description=self._metadata.get("description", ""),
            extra={"dataset_path": str(self._definition.path)},
        )

    def eval_df(self) -> pd.DataFrame:
        return self._eval.copy()

    def train_df(self) -> pd.DataFrame:
        return self._train.copy()

    def labels(self) -> list[bool]:
        return self._eval[self._label_column].astype(bool).tolist()


def get_handle(definition: DatasetDefinition) -> ExternalFraudDatasetHandle:
    return ExternalFraudDatasetHandle(definition)
