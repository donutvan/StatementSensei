from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from budget_sensei.storage import CONFIG_DIR

RULES_PATH = CONFIG_DIR / "rules.yaml"
CATEGORIES_PATH = CONFIG_DIR / "categories.yaml"


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


def add_category(name: str) -> list[str]:
    name = name.strip()
    if not name:
        return load_categories()
    categories = load_categories()
    if name not in categories:
        categories.append(name)
        with open(CATEGORIES_PATH, "w", encoding="utf-8") as file:
            yaml.safe_dump(categories, file)
    return categories


def categorize_rule_based(df: pd.DataFrame, rules: dict | None = None) -> pd.DataFrame:
    if df.empty:
        return df
    active_rules = rules or load_rules()
    if not active_rules:
        return df
    descriptions = df["description"].fillna("")
    for category, entries in active_rules.items():
        pattern = "|".join(entry.get("pattern", "") for entry in entries if entry.get("pattern"))
        if not pattern:
            continue
        mask = descriptions.str.contains(pattern, case=False, regex=True, na=False)
        df.loc[mask & df["category"].isin(["", "uncategorised", None]), "category"] = category
    return df


__all__ = ["load_rules", "load_categories", "add_category", "categorize_rule_based"]
