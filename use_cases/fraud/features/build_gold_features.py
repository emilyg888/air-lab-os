"""Deterministic builder for the fraud gold feature set."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from core.dataset_loader import load_dataset


REPO_ROOT = Path(__file__).resolve().parents[3]
GOLD_FEATURE_SET_DIR = Path(__file__).resolve().parent / "gold"
GOLD_FEATURE_SET_PATH = GOLD_FEATURE_SET_DIR / "gold_feature_set_v1.json"
GOLD_MAPPING_ARTIFACT_PATH = REPO_ROOT / "artifacts" / "gold_feature_set_v1_mappings.pkl"
GOLD_FEATURE_NAMES = ("merchant_rarity", "account_shared_merchant_count")


def _merchant_column(df: pd.DataFrame) -> str:
    return "merchant_id" if "merchant_id" in df.columns else "merchant_category"


def build_gold_mappings(train_df: pd.DataFrame) -> dict[str, Any]:
    merchant_col = _merchant_column(train_df)
    return {
        "merchant_column": merchant_col,
        "merchant_freq": train_df[merchant_col].value_counts().to_dict(),
        "account_merchants": (
            train_df.groupby("account_id")[merchant_col]
            .apply(lambda values: set(values.dropna().astype(str)))
            .to_dict()
        ),
    }


def build_gold_features(df: pd.DataFrame, mappings: dict[str, Any]) -> pd.DataFrame:
    enriched = df.copy()
    merchant_col = mappings.get("merchant_column") or _merchant_column(enriched)
    merchant_freq = mappings["merchant_freq"]
    account_merchants = mappings["account_merchants"]

    enriched["merchant_rarity"] = enriched[merchant_col].map(
        lambda value: 1.0 / (merchant_freq.get(value, 0) + 1.0)
    ).astype(float)
    enriched["account_shared_merchant_count"] = enriched["account_id"].map(
        lambda account_id: float(len(account_merchants.get(account_id, set())))
    ).astype(float)
    return enriched


def save_gold_mappings(
    mappings: dict[str, Any],
    path: Path = GOLD_MAPPING_ARTIFACT_PATH,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(mappings, handle)
    return path


def load_gold_mappings(path: Path = GOLD_MAPPING_ARTIFACT_PATH) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_gold_feature_set(path: Path = GOLD_FEATURE_SET_PATH) -> dict[str, Any]:
    return json.loads(path.read_text())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and save train-only mappings for the fraud gold feature set",
    )
    parser.add_argument(
        "--dataset",
        default="use_cases.fraud.handle",
        help="Dataset id or DatasetHandle module path",
    )
    parser.add_argument(
        "--output",
        default=str(GOLD_MAPPING_ARTIFACT_PATH),
        help="Output pickle path for saved mappings",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    handle = load_dataset(args.dataset)
    mappings = build_gold_mappings(handle.train_df())
    output = save_gold_mappings(mappings, Path(args.output))
    print(output)


if __name__ == "__main__":
    main()
