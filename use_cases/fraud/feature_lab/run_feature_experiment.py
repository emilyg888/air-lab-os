"""Run minimal one-feature-at-a-time experiments for fraud signals."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.dataset_loader import load_dataset
from datasets.base import DatasetHandle
from use_cases.fraud.features.build_gold_features import (
    GOLD_FEATURE_NAMES,
    GOLD_FEATURE_SET_PATH,
    build_gold_features,
    build_gold_mappings,
    load_gold_feature_set,
)
from use_cases.fraud.features.feature_contribution_log import (
    FEATURE_CONTRIBUTION_LOG_PATH,
    load_feature_contribution_log,
    save_feature_contribution_log,
)
from use_cases.fraud.features.feature_defs import (
    FEATURES,
    GRAPH_FEATURE_PACK_ALL,
    GRAPH_FEATURE_PACK_V1,
    GRAPH_FEATURE_PACK_V2,
)
from use_cases.fraud.patterns.ml_logistic import FEATURE_COLS as BASE_FEATURE_COLS


@dataclass(frozen=True)
class FeatureExperimentResult:
    feature_name: str
    baseline_f1: float
    experiment_f1: float
    delta_f1: float
    feature_count: int
    promotion_status: str | None = None


@dataclass(frozen=True)
class ModelExperimentResult:
    model_name: str
    baseline_f1: float
    experiment_f1: float
    delta_f1: float
    feature_count: int


@dataclass(frozen=True)
class FeaturePackExperimentResult:
    pack_name: str
    feature_names: tuple[str, ...]
    baseline_f1: float
    experiment_f1: float
    delta_f1: float
    feature_count: int


GRAPH_FEATURE_NAMES = set(GRAPH_FEATURE_PACK_ALL)
GOLD_FEATURE_SET = set(GOLD_FEATURE_NAMES)
DEFAULT_BASELINE_NAME = "default"
GOLD_BASELINE_NAME = "gold_feature_set_v1"


def _feature_columns(
    train_df: pd.DataFrame,
    extra_features: tuple[str, ...] = (),
    baseline_features: tuple[str, ...] = (),
) -> list[str]:
    cols = [name for name in BASE_FEATURE_COLS if name in train_df.columns]
    cols.extend(baseline_features)
    cols.extend(extra_features)
    return list(dict.fromkeys(cols))


def resolve_baseline_features(baseline_name: str = DEFAULT_BASELINE_NAME) -> tuple[str, ...]:
    if baseline_name == DEFAULT_BASELINE_NAME:
        return ()
    if baseline_name == GOLD_BASELINE_NAME:
        manifest = load_gold_feature_set()
        return tuple(manifest["features"])
    raise ValueError(
        f"Unknown baseline '{baseline_name}'. Expected one of: "
        f"{DEFAULT_BASELINE_NAME}, {GOLD_BASELINE_NAME}."
    )


def classify_feature_promotion(
    delta_f1: float,
    baseline_name: str = DEFAULT_BASELINE_NAME,
) -> str | None:
    if baseline_name != GOLD_BASELINE_NAME:
        return None
    if delta_f1 > 0.01:
        return "promote_to_gold_v2"
    if delta_f1 > 0.005:
        return "candidate"
    return "reject"


def update_feature_contribution_log(
    results: list[FeatureExperimentResult],
    baseline_name: str,
    path: Path = FEATURE_CONTRIBUTION_LOG_PATH,
) -> Path:
    log = load_feature_contribution_log(path)
    if baseline_name != GOLD_BASELINE_NAME:
        return save_feature_contribution_log(log, path)

    for result in results:
        if result.promotion_status is None:
            continue
        log[result.feature_name] = {
            "delta_f1": round(result.delta_f1, 4),
            "status": result.promotion_status,
        }
    return save_feature_contribution_log(log, path)


def _apply_standard_features(df: pd.DataFrame, feature_names: tuple[str, ...] = ()) -> pd.DataFrame:
    enriched = df.copy()
    for name in feature_names:
        enriched[name] = FEATURES[name](enriched)
    return enriched


def _split_feature_names(
    feature_names: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    standard: list[str] = []
    graph: list[str] = []
    gold: list[str] = []
    for name in feature_names:
        if name in GOLD_FEATURE_SET:
            gold.append(name)
        elif name in GRAPH_FEATURE_NAMES:
            graph.append(name)
        else:
            standard.append(name)
    return tuple(standard), tuple(graph), tuple(gold)


def _merchant_column(df: pd.DataFrame) -> str:
    return "merchant_id" if "merchant_id" in df.columns else "merchant_category"


def _safe_map(
    series: pd.Series,
    mapping: pd.Series,
    default: float,
) -> pd.Series:
    if mapping.empty:
        return pd.Series(default, index=series.index, dtype=float)
    return series.map(mapping).fillna(default).astype(float)


def _apply_graph_features(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_names: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not feature_names:
        return train_df.copy(), eval_df.copy()

    train = train_df.copy()
    eval_ = eval_df.copy()
    merchant_col = _merchant_column(train)

    merchant_account_map = train.groupby(merchant_col)["account_id"].nunique().astype(float)
    merchant_fraud_rate_map = train.groupby(merchant_col)["fraud_flag"].mean().astype(float)
    account_merchant_diversity_map = train.groupby("account_id")[merchant_col].nunique().astype(float)

    merchant_account_default = float(merchant_account_map.median()) if not merchant_account_map.empty else 0.0
    merchant_fraud_default = float(train["fraud_flag"].mean()) if len(train) else 0.0
    account_diversity_default = (
        float(account_merchant_diversity_map.median())
        if not account_merchant_diversity_map.empty
        else 0.0
    )

    if "merchant_account_count" in feature_names:
        train["merchant_account_count"] = _safe_map(train[merchant_col], merchant_account_map, merchant_account_default)
        eval_["merchant_account_count"] = _safe_map(eval_[merchant_col], merchant_account_map, merchant_account_default)

    if "merchant_fraud_rate" in feature_names or any(name in feature_names for name in GRAPH_FEATURE_PACK_V2):
        train["merchant_fraud_rate"] = _safe_map(train[merchant_col], merchant_fraud_rate_map, merchant_fraud_default)
        eval_["merchant_fraud_rate"] = _safe_map(eval_[merchant_col], merchant_fraud_rate_map, merchant_fraud_default)

    if "account_merchant_diversity" in feature_names:
        train["account_merchant_diversity"] = _safe_map(
            train["account_id"],
            account_merchant_diversity_map,
            account_diversity_default,
        )
        eval_["account_merchant_diversity"] = _safe_map(
            eval_["account_id"],
            account_merchant_diversity_map,
            account_diversity_default,
        )

    if any(name in feature_names for name in GRAPH_FEATURE_PACK_V2):
        account_exposure_map = train.groupby("account_id")["merchant_fraud_rate"].mean().astype(float)
        high_risk_ratio_map = (
            train.assign(_high_risk_flag=(train["merchant_fraud_rate"] > 0.5).astype(float))
            .groupby("account_id")["_high_risk_flag"]
            .mean()
            .astype(float)
        )
        account_exposure_default = float(train["merchant_fraud_rate"].mean()) if len(train) else 0.0
        high_risk_ratio_default = float((train["merchant_fraud_rate"] > 0.5).mean()) if len(train) else 0.0

        if "account_exposure_to_risk" in feature_names or "relative_merchant_risk" in feature_names:
            train["account_exposure_to_risk"] = _safe_map(
                train["account_id"],
                account_exposure_map,
                account_exposure_default,
            )
            eval_["account_exposure_to_risk"] = _safe_map(
                eval_["account_id"],
                account_exposure_map,
                account_exposure_default,
            )

        if "high_risk_merchant_ratio" in feature_names:
            train["high_risk_merchant_ratio"] = _safe_map(
                train["account_id"],
                high_risk_ratio_map,
                high_risk_ratio_default,
            )
            eval_["high_risk_merchant_ratio"] = _safe_map(
                eval_["account_id"],
                high_risk_ratio_map,
                high_risk_ratio_default,
            )

        if "relative_merchant_risk" in feature_names:
            train["relative_merchant_risk"] = (
                train["merchant_fraud_rate"] - train["account_exposure_to_risk"]
            )
            eval_["relative_merchant_risk"] = (
                eval_["merchant_fraud_rate"] - eval_["account_exposure_to_risk"]
            )

    return train, eval_


def _apply_gold_features(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_names: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not feature_names:
        return train_df.copy(), eval_df.copy()

    mappings = build_gold_mappings(train_df)
    train = build_gold_features(train_df, mappings)
    eval_ = build_gold_features(eval_df, mappings)
    return train, eval_


def _prepare_model_inputs(
    handle: DatasetHandle,
    extra_features: tuple[str, ...] = (),
    baseline_features: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    standard_features, graph_features, gold_features = _split_feature_names(extra_features)
    _, _, baseline_gold_features = _split_feature_names(baseline_features)
    train_df = handle.train_df()
    eval_df = handle.eval_df()
    train_df = _apply_standard_features(train_df, standard_features)
    eval_df = _apply_standard_features(eval_df, standard_features)
    train_df, eval_df = _apply_gold_features(
        train_df,
        eval_df,
        tuple(dict.fromkeys(baseline_gold_features + gold_features)),
    )
    train_df, eval_df = _apply_graph_features(train_df, eval_df, graph_features)
    feature_cols = _feature_columns(train_df, extra_features, baseline_features)
    return train_df, eval_df, feature_cols


def train_and_score(
    handle: DatasetHandle,
    extra_features: tuple[str, ...] = (),
    baseline_features: tuple[str, ...] = (),
    *,
    threshold: float = 0.5,
    max_iter: int = 1000,
    C: float = 1.0,
    class_weight: str = "balanced",
) -> tuple[float, int]:
    train_df, eval_df, feature_cols = _prepare_model_inputs(
        handle,
        extra_features,
        baseline_features,
    )

    X_train = train_df[feature_cols].fillna(0.0).values
    y_train = train_df["fraud_flag"].astype(int).values
    X_eval = eval_df[feature_cols].fillna(0.0).values
    y_eval = eval_df["fraud_flag"].astype(int).values

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                    class_weight=class_weight,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_eval)[:, 1]
    predictions = probabilities >= threshold
    return float(f1_score(y_eval, predictions, zero_division=0)), len(feature_cols)


def train_and_score_random_forest(
    handle: DatasetHandle,
    extra_features: tuple[str, ...] = (),
    baseline_features: tuple[str, ...] = (),
    *,
    n_estimators: int = 100,
    max_depth: int | None = None,
) -> tuple[float, int]:
    train_df, eval_df, feature_cols = _prepare_model_inputs(
        handle,
        extra_features,
        baseline_features,
    )

    X_train = train_df[feature_cols].fillna(0.0).values
    y_train = train_df["fraud_flag"].astype(int).values
    X_eval = eval_df[feature_cols].fillna(0.0).values
    y_eval = eval_df["fraud_flag"].astype(int).values

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_eval)
    return float(f1_score(y_eval, predictions, zero_division=0)), len(feature_cols)


def run_feature_experiments(
    handle: DatasetHandle,
    feature_names: tuple[str, ...] | None = None,
    baseline_features: tuple[str, ...] = (),
    baseline_name: str = DEFAULT_BASELINE_NAME,
) -> list[FeatureExperimentResult]:
    selected = tuple(
        name for name in (feature_names or tuple(FEATURES.keys()))
        if name not in baseline_features
    )
    baseline_f1, _ = train_and_score(handle, baseline_features=baseline_features)

    results: list[FeatureExperimentResult] = []
    for feature_name in selected:
        experiment_f1, feature_count = train_and_score(
            handle,
            (feature_name,),
            baseline_features=baseline_features,
        )
        results.append(
            FeatureExperimentResult(
                feature_name=feature_name,
                baseline_f1=baseline_f1,
                experiment_f1=experiment_f1,
                delta_f1=experiment_f1 - baseline_f1,
                feature_count=feature_count,
                promotion_status=classify_feature_promotion(
                    experiment_f1 - baseline_f1,
                    baseline_name,
                ),
            )
        )
    return results


def run_random_forest_experiment(
    handle: DatasetHandle,
    extra_features: tuple[str, ...] = (),
    baseline_features: tuple[str, ...] = (),
) -> ModelExperimentResult:
    baseline_f1, _ = train_and_score(handle, baseline_features=baseline_features)
    experiment_f1, feature_count = train_and_score_random_forest(
        handle,
        extra_features,
        baseline_features=baseline_features,
    )
    return ModelExperimentResult(
        model_name="random_forest",
        baseline_f1=baseline_f1,
        experiment_f1=experiment_f1,
        delta_f1=experiment_f1 - baseline_f1,
        feature_count=feature_count,
    )


def run_feature_pack_experiment(
    handle: DatasetHandle,
    pack_name: str,
    feature_names: tuple[str, ...],
    baseline_features: tuple[str, ...] = (),
) -> FeaturePackExperimentResult:
    baseline_f1, _ = train_and_score(handle, baseline_features=baseline_features)
    experiment_f1, feature_count = train_and_score(
        handle,
        feature_names,
        baseline_features=baseline_features,
    )
    return FeaturePackExperimentResult(
        pack_name=pack_name,
        feature_names=feature_names,
        baseline_f1=baseline_f1,
        experiment_f1=experiment_f1,
        delta_f1=experiment_f1 - baseline_f1,
        feature_count=feature_count,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one-feature-at-a-time fraud feature experiments",
    )
    parser.add_argument(
        "--dataset",
        default="use_cases.fraud.handle",
        help="Dataset id or DatasetHandle module path",
    )
    parser.add_argument(
        "--baseline",
        default=DEFAULT_BASELINE_NAME,
        choices=(DEFAULT_BASELINE_NAME, GOLD_BASELINE_NAME),
        help=(
            "Baseline feature set to compare against. "
            f"'{GOLD_BASELINE_NAME}' loads features from {GOLD_FEATURE_SET_PATH}."
        ),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    handle = load_dataset(args.dataset)
    baseline_features = resolve_baseline_features(args.baseline)
    results = run_feature_experiments(
        handle,
        baseline_features=baseline_features,
        baseline_name=args.baseline,
    )

    if not results:
        print("No features configured.")
        return

    baseline_f1 = results[0].baseline_f1
    contribution_log_path: Path | None = None
    if args.baseline == GOLD_BASELINE_NAME:
        contribution_log_path = update_feature_contribution_log(results, args.baseline)

    print(f"Dataset: {handle.meta.name}")
    print(f"Baseline: {args.baseline}")
    print(f"Baseline F1: {baseline_f1:.4f}")
    print("")

    for result in results:
        delta = f"{result.delta_f1:+.4f}"
        print(f"Feature: {result.feature_name}")
        print(f"New F1: {result.experiment_f1:.4f}")
        print(f"Delta F1: {delta}")
        if result.promotion_status is not None:
            print(f"Promotion: {result.promotion_status}")
        print("")

    if contribution_log_path is not None:
        print(f"Contribution log: {contribution_log_path}")


if __name__ == "__main__":
    main()
