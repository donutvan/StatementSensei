"""Persistence helpers for Statement Sensei."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
import pandas as pd
import streamlit as st

APP_DIR = Path.home() / ".statement_sensei"
TRANSACTION_DB_PATH = APP_DIR / "transactions.db"
OVERRIDES_PATH = APP_DIR / "category_overrides.json"


def _ensure_app_dir() -> None:
    try:
        APP_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        st.warning(f"Unable to prepare data directory {APP_DIR}: {exc}")


def persist_transactions(df: pd.DataFrame) -> None:
    """Store the latest normalized dataframe inside a lightweight SQLite DB."""

    if df.empty:
        return

    _ensure_app_dir()
    try:
        with sqlite3.connect(TRANSACTION_DB_PATH) as conn:
            df.to_sql("transactions", conn, if_exists="replace", index=False)
    except Exception as exc:  # pylint: disable=broad-except
        st.warning(f"Unable to persist transactions to {TRANSACTION_DB_PATH}: {exc}")


def load_overrides() -> dict[str, str]:
    """Read user provided category overrides from disk."""

    if not OVERRIDES_PATH.exists():
        return {}

    try:
        return json.loads(OVERRIDES_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        st.warning(f"Failed to read overrides from {OVERRIDES_PATH}: {exc}")
        return {}


def upsert_overrides(updates: dict[str, str]) -> dict[str, str]:
    """Merge new overrides with the stored mapping."""

    if not updates:
        return load_overrides()

    overrides = load_overrides()
    overrides.update({k: v for k, v in updates.items() if v})
    _ensure_app_dir()
    try:
        OVERRIDES_PATH.write_text(json.dumps(overrides, indent=2, sort_keys=True))
    except OSError as exc:
        st.error(f"Unable to store overrides: {exc}")
    return overrides
