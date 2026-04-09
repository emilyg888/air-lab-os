from types import SimpleNamespace

import pandas as pd
import pytest

from core.dataset_loader import AliasedDatasetHandle, load_dataset
from core.evaluation import evaluate
from core.registry import load_registry, update_registry
from datasets.base import DatasetHandle, DatasetMeta
from datasets.registry import DatasetDefinition
from patterns.base import RunResult
from use_cases.fraud.registry_handle import get_handle as get_registry_fraud_handle


class StubHandle(DatasetHandle):
    @property
    def meta(self):
        return DatasetMeta(
            name="stub",
            domain="test",
            tier="test",
            version="0.1",
            label_column="label",
            primary_metric="f1_score",
            row_count=4,
        )

    def eval_df(self):
        return pd.DataFrame({"label": [True, True, False, False]})

    def train_df(self):
        return self.eval_df()

    def labels(self):
        return [True, True, False, False]


def test_evaluate_accepts_raw_policy_dict_for_dataset_handle():
    result = RunResult(
        flags=[True, True, False, False],
        scores=[0.9, 0.8, 0.1, 0.2],
        explanation=["x", "y", "", ""],
    )
    policy = {
        "weights": {
            "primary_metric": {"weight": 0.60},
            "explainability": {"weight": 0.20},
            "latency": {"weight": 0.10},
            "cost": {"weight": 0.10},
        },
        "constraints": {
            "latency": {"max_ms": 500},
            "cost": {"max_per_1k": 0.10},
        },
        "rules": {
            "primary_metric_floor": 0.1,
            "stability_threshold": 0.05,
            "penalty_factor": 0.5,
            "stability_gap_threshold": 0.2,
            "promotion_confidence_threshold": 0.7,
        },
        "arena": {
            "ranking": {"stable_bonus": 1.1, "unstable_bonus": 0.9},
            "exploration": {"max_runs_per_pattern": 5},
        },
        "promotion": {
            "working_threshold": 0.55,
            "stable_threshold": 0.72,
            "min_runs": 3,
        },
    }

    metrics = evaluate(result, StubHandle(), policy)

    assert metrics.primary_metric_value == pytest.approx(1.0)
    assert 0.0 <= metrics.score <= 1.0


def _write_dataset_package(
    tmp_path,
    *,
    dataset_id: str,
    data_csv: str,
    metadata_json: str,
    labels_csv: str | None = None,
    domain: str = "fraud",
    tier: str = "test",
    version: str = "1.0",
    has_labels: bool = True,
):
    dataset_dir = tmp_path / dataset_id
    dataset_dir.mkdir()
    (dataset_dir / "data.csv").write_text(data_csv)
    (dataset_dir / "metadata.json").write_text(metadata_json)
    if labels_csv is not None:
        (dataset_dir / "labels.csv").write_text(labels_csv)

    labels_path = dataset_dir / "labels.csv"
    return DatasetDefinition(
        dataset_id=dataset_id,
        domain=domain,
        path=dataset_dir,
        version=version,
        tier=tier,
        has_labels=has_labels,
        metadata_path=dataset_dir / "metadata.json",
        schema_path=None,
        data_path=dataset_dir / "data.csv",
        labels_path=labels_path if labels_path.exists() else None,
    )


def test_load_dataset_registry_normalizes_small_fraud_dataset(tmp_path, monkeypatch):
    definition = _write_dataset_package(
        tmp_path,
        dataset_id="synthetic_fraud",
        data_csv=(
            "transaction_id,account_id,amount,timestamp,scenario,severity\n"
            "sim_001,acct_s01,9999.00,2024-04-01T00:00:00,burst,high\n"
            "sim_002,acct_s01,9999.00,2024-04-01T00:00:01,burst,high\n"
            "sim_003,acct_s02,20.00,2024-04-01T00:10:00,safe,low\n"
        ),
        metadata_json=(
            "{"
            "\"name\": \"synthetic_fraud\", "
            "\"domain\": \"fraud\", "
            "\"description\": \"Synthetic fraud scenarios.\", "
            "\"label_column\": \"is_fraud\", "
            "\"primary_keys\": [\"transaction_id\"], "
            "\"evaluation_metric\": \"f1_score\""
            "}"
        ),
        labels_csv=(
            "transaction_id,is_fraud\n"
            "sim_001,true\n"
            "sim_002,true\n"
            "sim_003,false\n"
        ),
    )
    monkeypatch.setattr("core.dataset_loader.get_dataset_definition", lambda name: definition)

    handle = load_dataset("synthetic_fraud")

    assert handle.meta.name == "synthetic_fraud"
    assert handle.meta.label_column == "fraud_flag"
    assert handle.train_df().equals(handle.eval_df())
    assert handle.labels() == [True, True, False]
    assert {"txn_id", "txn_type", "fraud_flag", "same_ts_count"}.issubset(handle.eval_df().columns)


def test_load_dataset_falls_back_to_module_handle(monkeypatch):
    stub_module = SimpleNamespace(get_handle=lambda: StubHandle())

    def _missing(name):
        raise KeyError(name)

    monkeypatch.setattr("core.dataset_loader.get_dataset_definition", _missing)
    monkeypatch.setattr("core.dataset_loader.importlib.import_module", lambda name: stub_module)

    handle = load_dataset("custom.handle")

    assert isinstance(handle, AliasedDatasetHandle)
    assert handle.meta.name == "custom.handle"
    assert handle.labels() == [True, True, False, False]


def test_edge_case_registry_handle_builds_fraud_compatible_frame(tmp_path):
    definition = _write_dataset_package(
        tmp_path,
        dataset_id="edge_cases_v1",
        data_csv=(
            "case_id,description,expected_signal,severity\n"
            "edge_001,Single very large debit,spike,high\n"
            "edge_002,Five same-second debits,velocity,high\n"
            "edge_003,Normal recurring payment,none,low\n"
        ),
        metadata_json=(
            "{"
            "\"name\": \"edge_cases_v1\", "
            "\"domain\": \"synthetic\", "
            "\"description\": \"Edge-case scenarios.\", "
            "\"primary_keys\": [\"case_id\"], "
            "\"evaluation_metric\": \"coverage\""
            "}"
        ),
        labels_csv=None,
        domain="synthetic",
        has_labels=False,
    )

    handle = get_registry_fraud_handle(definition)

    assert handle.meta.domain == "fraud"
    assert handle.meta.primary_metric == "f1_score"
    assert handle.train_df().equals(handle.eval_df())
    assert any(handle.labels())
    assert any(not label for label in handle.labels())
    assert {"txn_id", "account_id", "txn_type", "fraud_flag"}.issubset(handle.eval_df().columns)


def test_update_registry_backfills_legacy_entry_without_scores(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        """
{
  "rule_spike": {
    "runs": 1,
    "confidence": 0.6162,
    "last_score": 0.6162,
    "status": "bronze",
    "is_stable": false,
    "last_updated": "2026-04-09T10:00:00"
  }
}
""".strip()
    )

    updated = update_registry(
        pattern_name="rule_spike",
        score=0.7,
        metadata={},
        policy={
            "promotion": {
                "working_threshold": 0.65,
                "stable_threshold": 0.78,
                "min_runs": 3,
            },
            "rules": {
                "stability_threshold": 0.05,
                "promotion_confidence_threshold": 0.7,
            },
        },
        path=registry_path,
    )

    assert updated["runs"] == 2
    assert updated["scores"] == [0.6162, 0.7]
    assert load_registry(registry_path)["rule_spike"]["last_score"] == pytest.approx(0.7)


def test_update_registry_uses_policy_stability_threshold(tmp_path):
    registry_path = tmp_path / "registry.json"
    policy = {
        "promotion": {
            "working_threshold": 0.55,
            "stable_threshold": 0.72,
            "min_runs": 3,
        },
        "rules": {
            "stability_threshold": 0.05,
            "promotion_confidence_threshold": 0.7,
        },
    }

    update_registry("pattern_a", 0.55, {}, policy, path=registry_path)
    update_registry("pattern_a", 0.80, {}, policy, path=registry_path)
    updated = update_registry("pattern_a", 0.65, {}, policy, path=registry_path)

    assert updated["is_stable"] is True
