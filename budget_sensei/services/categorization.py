from __future__ import annotations

import pandas as pd

from budget_sensei.ml.inference import predict_categories
from budget_sensei.rules import categorize_rule_based
from budget_sensei.storage import load_overrides, load_transactions, update_categories_bulk


def _apply_overrides(df: pd.DataFrame) -> pd.DataFrame:
    overrides = load_overrides()
    if overrides.empty:
        return df
    override_map = dict(zip(overrides["transaction_hash"], overrides["category"]))
    df.loc[:, "category"] = df["transaction_hash"].map(override_map).combine_first(df["category"])
    return df


def categorize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = _apply_overrides(df)
    df = categorize_rule_based(df)
    df = predict_categories(df)
    df["category"] = df["category"].fillna("uncategorised")
    return df


def recategorize_all_transactions() -> int:
    transactions = load_transactions()
    if transactions.empty:
        return 0
    updated = categorize_transactions(transactions.copy())
    return update_categories_bulk(updated[["transaction_hash", "category"]])


__all__ = ["categorize_transactions", "recategorize_all_transactions"]
