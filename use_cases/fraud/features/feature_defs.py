"""Deterministic candidate signals for fraud feature experiments."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd


FeatureFn = Callable[[pd.DataFrame], pd.Series]

GRAPH_FEATURE_PACK_V1 = (
    "merchant_account_count",
    "merchant_fraud_rate",
    "account_merchant_diversity",
)

GRAPH_FEATURE_PACK_V2 = (
    "account_exposure_to_risk",
    "high_risk_merchant_ratio",
    "relative_merchant_risk",
)

GRAPH_FEATURE_PACK_ALL = GRAPH_FEATURE_PACK_V1 + GRAPH_FEATURE_PACK_V2


def _amount_to_balance_ratio(df: pd.DataFrame) -> pd.Series:
    balance = df["account_balance"].abs().clip(lower=1.0)
    return df["abs_amount"].astype(float) / balance


def _same_ts_amount_pressure(df: pd.DataFrame) -> pd.Series:
    return df["same_ts_count"].clip(lower=1).astype(float) * df["abs_amount"].astype(float)


def _high_risk_debit_flag(df: pd.DataFrame) -> pd.Series:
    return (
        df["txn_type"].eq("DEBIT") & df["risk_rating"].eq("High")
    ).astype(float)


def _per_account_time_feature(
    df: pd.DataFrame,
    builder: Callable[[pd.DataFrame], pd.Series],
) -> pd.Series:
    working = df.copy()
    working["_source_index"] = working.index
    working["_feature_ts"] = pd.to_datetime(working["timestamp"], errors="coerce")
    sort_cols = ["account_id", "_feature_ts"]
    if "txn_id" in working.columns:
        sort_cols.append("txn_id")
    ordered = working.sort_values(sort_cols, kind="mergesort")

    parts: list[pd.Series] = []
    for _, group in ordered.groupby("account_id", sort=False):
        parts.append(builder(group))

    if not parts:
        return pd.Series(0.0, index=df.index, dtype=float)

    features = pd.concat(parts)
    return features.reindex(df.index).fillna(0.0)


def _timestamp_seconds(group: pd.DataFrame) -> list[float]:
    if group.empty:
        return []
    anchor = group["_feature_ts"].iloc[0]
    return (group["_feature_ts"] - anchor).dt.total_seconds().tolist()


def _window_starts(ts_seconds: list[float], window_seconds: float) -> list[int]:
    starts: list[int] = []
    left = 0
    for right, current in enumerate(ts_seconds):
        while left < right and current - ts_seconds[left] > window_seconds:
            left += 1
        starts.append(left)
    return starts


def _window_means(values: list[float], starts: list[int]) -> list[float]:
    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + value)

    means: list[float] = []
    for idx, start in enumerate(starts):
        count = idx - start + 1
        total = prefix[idx + 1] - prefix[start]
        means.append(total / count)
    return means


def _burst_density_10m(df: pd.DataFrame) -> pd.Series:
    def _builder(group: pd.DataFrame) -> pd.Series:
        ts_seconds = _timestamp_seconds(group)
        starts = _window_starts(ts_seconds, window_seconds=600.0)

        values: list[float] = []
        for idx, start in enumerate(starts):
            tx_count = idx - start + 1
            time_span = max(ts_seconds[idx] - ts_seconds[start], 0.0)
            values.append(tx_count / (time_span + 1.0))

        return pd.Series(values, index=group["_source_index"], dtype=float)

    return _per_account_time_feature(df, _builder)


def _behaviour_drift(df: pd.DataFrame) -> pd.Series:
    def _builder(group: pd.DataFrame) -> pd.Series:
        ts_seconds = _timestamp_seconds(group)
        amounts = group["amount"].astype(float).tolist()
        starts_1h = _window_starts(ts_seconds, window_seconds=3600.0)
        starts_7d = _window_starts(ts_seconds, window_seconds=604800.0)
        avg_1h = _window_means(amounts, starts_1h)
        avg_7d = _window_means(amounts, starts_7d)

        values = [
            abs(short - long) / (abs(long) + 1e-5)
            for short, long in zip(avg_1h, avg_7d)
        ]
        return pd.Series(values, index=group["_source_index"], dtype=float)

    return _per_account_time_feature(df, _builder)


def _behaviour_drift_v2(df: pd.DataFrame) -> pd.Series:
    def _builder(group: pd.DataFrame) -> pd.Series:
        ts_seconds = _timestamp_seconds(group)
        amounts = group["amount"].astype(float).tolist()
        starts_1h = _window_starts(ts_seconds, window_seconds=3600.0)
        starts_7d = _window_starts(ts_seconds, window_seconds=604800.0)
        avg_1h = _window_means(amounts, starts_1h)
        avg_7d = _window_means(amounts, starts_7d)

        values: list[float] = []
        for idx, (short, long) in enumerate(zip(avg_1h, avg_7d)):
            baseline_drift = abs(short - long) / (abs(long) + 1e-5)
            tx_count_1h = idx - starts_1h[idx] + 1
            frequency_weight = 1.0 + (tx_count_1h - 1) / tx_count_1h
            values.append(baseline_drift * frequency_weight)

        return pd.Series(values, index=group["_source_index"], dtype=float)

    return _per_account_time_feature(df, _builder)


def _gap_change(df: pd.DataFrame) -> pd.Series:
    def _builder(group: pd.DataFrame) -> pd.Series:
        ts_seconds = _timestamp_seconds(group)

        time_gaps: list[float | None] = [None]
        for idx in range(1, len(ts_seconds)):
            time_gaps.append(ts_seconds[idx] - ts_seconds[idx - 1])

        gap_changes: list[float] = [0.0]
        for idx in range(1, len(time_gaps)):
            current_gap = time_gaps[idx]
            previous_gap = time_gaps[idx - 1]
            if current_gap is None or previous_gap is None:
                gap_changes.append(0.0)
                continue
            gap_changes.append(current_gap - previous_gap)

        clipped = [min(3600.0, max(-3600.0, value)) for value in gap_changes]
        return pd.Series(clipped, index=group["_source_index"], dtype=float)

    return _per_account_time_feature(df, _builder)


def _merchant_account_count(df: pd.DataFrame) -> pd.Series:
    merchant_col = "merchant_id" if "merchant_id" in df.columns else "merchant_category"
    return (
        df.groupby(merchant_col)["account_id"]
        .transform("nunique")
        .astype(float)
    )


def _account_shared_merchant_count(df: pd.DataFrame) -> pd.Series:
    merchant_col = "merchant_id" if "merchant_id" in df.columns else "merchant_category"
    return (
        df.groupby("account_id")[merchant_col]
        .transform("nunique")
        .astype(float)
    )


def _merchant_rarity(df: pd.DataFrame) -> pd.Series:
    merchant_col = "merchant_id" if "merchant_id" in df.columns else "merchant_category"
    merchant_freq = df[merchant_col].value_counts()
    return df[merchant_col].map(lambda value: 1.0 / (merchant_freq[value] + 1.0)).astype(float)


def _account_max_merchant_exposure(df: pd.DataFrame) -> pd.Series:
    merchant_exposure = (
        df["merchant_account_count"]
        if "merchant_account_count" in df.columns
        else _merchant_account_count(df)
    )
    working = df.copy()
    working["_merchant_account_count"] = merchant_exposure
    return (
        working.groupby("account_id")["_merchant_account_count"]
        .transform("max")
        .astype(float)
    )


def _merchant_diversity(df: pd.DataFrame) -> pd.Series:
    def _builder(group: pd.DataFrame) -> pd.Series:
        ts_seconds = _timestamp_seconds(group)
        merchants = group["merchant_category"].fillna("Unknown").astype(str).tolist()
        starts = _window_starts(ts_seconds, window_seconds=3600.0)

        values: list[float] = []
        for idx, start in enumerate(starts):
            window = merchants[start : idx + 1]
            values.append(len(set(window)) / (len(window) + 1.0))

        return pd.Series(values, index=group["_source_index"], dtype=float)

    return _per_account_time_feature(df, _builder)


FEATURES: dict[str, FeatureFn] = {
    "amount_to_balance_ratio": _amount_to_balance_ratio,
    "same_ts_amount_pressure": _same_ts_amount_pressure,
    "high_risk_debit_flag": _high_risk_debit_flag,
    "burst_density_10m": _burst_density_10m,
    "behaviour_drift": _behaviour_drift,
    "behaviour_drift_v2": _behaviour_drift_v2,
    "gap_change": _gap_change,
    "merchant_account_count": _merchant_account_count,
    "account_shared_merchant_count": _account_shared_merchant_count,
    "merchant_rarity": _merchant_rarity,
    "account_max_merchant_exposure": _account_max_merchant_exposure,
    "merchant_diversity": _merchant_diversity,
}
