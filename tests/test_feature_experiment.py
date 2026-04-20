import pandas as pd
from pathlib import Path

from datasets.base import DatasetHandle, DatasetMeta
from use_cases.fraud.features.build_gold_features import (
    GOLD_FEATURE_NAMES,
    build_gold_features,
    build_gold_mappings,
    load_gold_feature_set,
)
from use_cases.fraud.feature_lab.run_feature_experiment import (
    GOLD_BASELINE_NAME,
    DEFAULT_BASELINE_NAME,
    FeatureExperimentResult,
    classify_feature_promotion,
    run_feature_experiments,
    run_feature_pack_experiment,
    run_random_forest_experiment,
    resolve_baseline_features,
    train_and_score,
    train_and_score_random_forest,
    update_feature_contribution_log,
)
from use_cases.fraud.features.feature_defs import (
    FEATURES,
    GRAPH_FEATURE_PACK_ALL,
    GRAPH_FEATURE_PACK_V1,
    GRAPH_FEATURE_PACK_V2,
)
from use_cases.fraud.features.feature_contribution_log import (
    load_feature_contribution_log,
    save_feature_contribution_log,
)


class StubFraudFeatureHandle(DatasetHandle):
    def __init__(self) -> None:
        base = pd.DataFrame(
            {
                "txn_id": [f"T{i:04d}" for i in range(10)],
                "account_id": ["A1", "A1", "A2", "A2", "A3", "A3", "A4", "A4", "A5", "A5"],
                "amount": [120.0, 180.0, 60.0, 240.0, 75.0, 400.0, 55.0, 520.0, 90.0, 610.0],
                "abs_amount": [120.0, 180.0, 60.0, 240.0, 75.0, 400.0, 55.0, 520.0, 90.0, 610.0],
                "same_ts_count": [1, 2, 1, 3, 1, 2, 1, 4, 1, 2],
                "z_score": [0.1, 0.2, -0.3, 0.5, -0.2, 1.1, -0.4, 1.5, -0.1, 1.8],
                "account_zscore": [0.0, 0.1, -0.2, 0.3, -0.1, 0.8, -0.2, 1.0, -0.1, 1.2],
                "is_burst": [False, True, False, True, False, True, False, True, False, True],
                "account_had_burst": [False, True, False, True, False, True, False, True, False, True],
                "account_balance": [1000.0, 900.0, 500.0, 400.0, 800.0, 350.0, 750.0, 300.0, 650.0, 250.0],
                "txn_type": ["DEBIT", "DEBIT", "CREDIT", "DEBIT", "CREDIT", "DEBIT", "CREDIT", "DEBIT", "CREDIT", "DEBIT"],
                "risk_rating": ["Low", "High", "Low", "High", "Low", "High", "Low", "High", "Low", "High"],
                "merchant_category": ["Food", "Fuel", "Food", "Travel", "Bills", "Retail", "Bills", "Gaming", "Bills", "Travel"],
                "timestamp": [
                    "2026-01-01T00:00:00",
                    "2026-01-01T00:03:00",
                    "2026-01-01T00:20:00",
                    "2026-01-01T00:25:00",
                    "2026-01-01T02:00:00",
                    "2026-01-01T02:05:00",
                    "2026-01-02T00:00:00",
                    "2026-01-02T00:07:00",
                    "2026-01-03T00:00:00",
                    "2026-01-03T00:09:00",
                ],
                "fraud_flag": [False, True, False, True, False, True, False, True, False, True],
            }
        )
        self._train = base.iloc[:8].reset_index(drop=True)
        self._eval = base.iloc[8:].reset_index(drop=True)

    @property
    def meta(self) -> DatasetMeta:
        return DatasetMeta(
            name="stub_fraud_features",
            domain="fraud",
            tier="test",
            version="0.1",
            label_column="fraud_flag",
            primary_metric="f1_score",
            row_count=len(self._train) + len(self._eval),
        )

    def eval_df(self) -> pd.DataFrame:
        return self._eval.copy()

    def train_df(self) -> pd.DataFrame:
        return self._train.copy()

    def labels(self) -> list[bool]:
        return self._eval["fraud_flag"].astype(bool).tolist()


def test_feature_defs_return_one_series_per_row():
    frame = StubFraudFeatureHandle().train_df()

    for name, feature_fn in FEATURES.items():
        series = feature_fn(frame)
        assert len(series) == len(frame), name
        assert series.notna().all(), name


def test_train_and_score_accepts_extra_feature():
    handle = StubFraudFeatureHandle()

    score, feature_count = train_and_score(handle, ("amount_to_balance_ratio",))

    assert 0.0 <= score <= 1.0
    assert feature_count == 8


def test_train_and_score_random_forest_runs_on_baseline_features():
    handle = StubFraudFeatureHandle()

    score, feature_count = train_and_score_random_forest(handle)

    assert 0.0 <= score <= 1.0
    assert feature_count == 7


def test_train_and_score_accepts_explicit_gold_baseline_features():
    handle = StubFraudFeatureHandle()

    score, feature_count = train_and_score(handle, baseline_features=GOLD_FEATURE_NAMES)

    assert 0.0 <= score <= 1.0
    assert feature_count == 9


def test_run_feature_experiments_returns_one_result_per_feature():
    handle = StubFraudFeatureHandle()

    results = run_feature_experiments(handle)

    assert [result.feature_name for result in results] == list(FEATURES.keys())
    assert all(result.baseline_f1 == results[0].baseline_f1 for result in results)


def test_run_feature_experiments_skips_gold_features_when_gold_is_baseline():
    handle = StubFraudFeatureHandle()

    results = run_feature_experiments(
        handle,
        baseline_features=GOLD_FEATURE_NAMES,
        baseline_name=GOLD_BASELINE_NAME,
    )

    assert "merchant_rarity" not in [result.feature_name for result in results]
    assert "account_shared_merchant_count" not in [result.feature_name for result in results]


def test_classify_feature_promotion_uses_gold_thresholds():
    assert classify_feature_promotion(0.011, GOLD_BASELINE_NAME) == "promote_to_gold_v2"
    assert classify_feature_promotion(0.006, GOLD_BASELINE_NAME) == "candidate"
    assert classify_feature_promotion(0.005, GOLD_BASELINE_NAME) == "reject"
    assert classify_feature_promotion(0.02, DEFAULT_BASELINE_NAME) is None


def test_gap_change_captures_acceleration_and_clips_outliers():
    frame = pd.DataFrame(
        {
            "txn_id": ["g1", "g2", "g3", "g4", "g5"],
            "account_id": ["A1"] * 5,
            "timestamp": [
                "2026-01-01T00:00:00",
                "2026-01-01T00:10:00",
                "2026-01-01T00:15:00",
                "2026-01-01T00:16:00",
                "2026-01-01T03:16:00",
            ],
        }
    )

    values = FEATURES["gap_change"](frame).tolist()

    assert values == [0.0, 0.0, -300.0, -240.0, 3600.0]


def test_merchant_account_count_varies_by_merchant():
    frame = StubFraudFeatureHandle().train_df()

    values = FEATURES["merchant_account_count"](frame)

    assert values.nunique() > 1
    assert values.max() > values.min()


def test_new_graph_v2_features_have_variation():
    frame = pd.DataFrame(
        {
            "account_id": ["A1", "A1", "A1", "A2", "A2", "A3"],
            "merchant_category": ["Food", "Fuel", "Bills", "Food", "Food", "Travel"],
        }
    )

    shared_count = FEATURES["account_shared_merchant_count"](frame)
    merchant_rarity = FEATURES["merchant_rarity"](frame)
    merchant_count_frame = frame.copy()
    merchant_count_frame["merchant_account_count"] = FEATURES["merchant_account_count"](merchant_count_frame)
    max_exposure = FEATURES["account_max_merchant_exposure"](merchant_count_frame)

    assert shared_count.nunique() > 1
    assert merchant_rarity.nunique() > 1
    assert max_exposure.nunique() > 1


def test_run_random_forest_experiment_returns_delta_against_logistic_baseline():
    handle = StubFraudFeatureHandle()

    result = run_random_forest_experiment(handle)

    assert result.model_name == "random_forest"
    assert 0.0 <= result.experiment_f1 <= 1.0


def test_gold_feature_builder_uses_train_only_mappings():
    train_df = StubFraudFeatureHandle().train_df()
    eval_df = StubFraudFeatureHandle().eval_df()

    mappings = build_gold_mappings(train_df)
    built = build_gold_features(eval_df, mappings)

    assert GOLD_FEATURE_NAMES == ("merchant_rarity", "account_shared_merchant_count")
    assert {"merchant_rarity", "account_shared_merchant_count"}.issubset(built.columns)
    assert built["account_shared_merchant_count"].notna().all()


def test_gold_baseline_manifest_and_runner_resolution_match():
    manifest = load_gold_feature_set()

    assert manifest["name"] == GOLD_BASELINE_NAME
    assert tuple(manifest["features"]) == GOLD_FEATURE_NAMES
    assert resolve_baseline_features(GOLD_BASELINE_NAME) == GOLD_FEATURE_NAMES


def test_update_feature_contribution_log_preserves_seeded_gold_entries(tmp_path: Path):
    log_path = tmp_path / "feature_contribution_log.json"
    save_feature_contribution_log(
        {
            "merchant_rarity": {"delta_f1": 0.0068, "status": "gold"},
            "account_shared_merchant_count": {"delta_f1": 0.0041, "status": "gold"},
        },
        log_path,
    )

    update_feature_contribution_log(
        [
            FeatureExperimentResult(
                feature_name="gap_change",
                baseline_f1=0.8656,
                experiment_f1=0.8770,
                delta_f1=0.0114,
                feature_count=10,
                promotion_status="promote_to_gold_v2",
            )
        ],
        GOLD_BASELINE_NAME,
        log_path,
    )
    log = load_feature_contribution_log(log_path)

    assert log["merchant_rarity"]["status"] == "gold"
    assert log["account_shared_merchant_count"]["status"] == "gold"
    assert log["gap_change"] == {
        "delta_f1": 0.0114,
        "status": "promote_to_gold_v2",
    }


def test_run_feature_pack_experiment_returns_pack_result():
    handle = StubFraudFeatureHandle()

    result = run_feature_pack_experiment(handle, "v1", GRAPH_FEATURE_PACK_V1)

    assert result.pack_name == "v1"
    assert result.feature_names == GRAPH_FEATURE_PACK_V1
    assert 0.0 <= result.experiment_f1 <= 1.0


def test_graph_feature_pack_definitions_match_expected_sequence():
    assert GRAPH_FEATURE_PACK_V1 == (
        "merchant_account_count",
        "merchant_fraud_rate",
        "account_merchant_diversity",
    )
    assert GRAPH_FEATURE_PACK_V2 == (
        "account_exposure_to_risk",
        "high_risk_merchant_ratio",
        "relative_merchant_risk",
    )
    assert GRAPH_FEATURE_PACK_ALL == GRAPH_FEATURE_PACK_V1 + GRAPH_FEATURE_PACK_V2


def test_feature_catalog_includes_twelve_candidates():
    assert list(FEATURES.keys()) == [
        "amount_to_balance_ratio",
        "same_ts_amount_pressure",
        "high_risk_debit_flag",
        "burst_density_10m",
        "behaviour_drift",
        "behaviour_drift_v2",
        "gap_change",
        "merchant_account_count",
        "account_shared_merchant_count",
        "merchant_rarity",
        "account_max_merchant_exposure",
        "merchant_diversity",
    ]
