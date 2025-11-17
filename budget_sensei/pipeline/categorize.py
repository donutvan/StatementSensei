from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import yaml

from .utils import CONFIG_DIR, load_overrides

RULES_PATH = CONFIG_DIR / "rules.yaml"
CATEGORIES_PATH = CONFIG_DIR / "categories.yaml"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_rules() -> dict:
    if not RULES_PATH.exists():
        return {}
    with open(RULES_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_categories() -> list[str]:
    if not CATEGORIES_PATH.exists():
        return []
    with open(CATEGORIES_PATH, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or []
    return list(data)


def _apply_overrides(df: pd.DataFrame) -> pd.DataFrame:
    overrides = load_overrides()
    if overrides.empty:
        return df
    override_map = dict(zip(overrides["transaction_hash"], overrides["category"]))
    df.loc[:, "category"] = df["transaction_hash"].map(override_map).combine_first(df["category"])
    return df


def _apply_rules(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    if not rules:
        return df
    descriptions = df["description"].fillna("")
    for category, entries in rules.items():
        pattern = "|".join(entry.get("pattern", "") for entry in entries if entry.get("pattern"))
        if not pattern:
            continue
        mask = descriptions.str.contains(pattern, case=False, regex=True, na=False)
        df.loc[mask & df["category"].isin(["", "uncategorised", None]), "category"] = category
    return df


def _load_model():
    vectorizer_path = DATA_DIR / "vectorizer.pkl"
    model_path = DATA_DIR / "model.pkl"
    if not vectorizer_path.exists() or not model_path.exists():
        return None, None
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    return vectorizer, model


def _apply_ml(df: pd.DataFrame) -> pd.DataFrame:
    vectorizer, model = _load_model()
    if vectorizer is None or model is None:
        return df
    mask = df["category"].isin(["", "uncategorised", None])
    if not mask.any():
        return df
    text = (df.loc[mask, "description"].fillna("") + " " + df.loc[mask, "amount"].abs().astype(str)).values
    features = vectorizer.transform(text)
    predictions = model.predict(features)
    df.loc[mask, "category"] = predictions
    return df


def categorize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = _apply_overrides(df)
    df = _apply_rules(df, load_rules())
    df = _apply_ml(df)
    df["category"] = df["category"].fillna("uncategorised")
    return df
