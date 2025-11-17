from __future__ import annotations

import pandas as pd
import streamlit as st

from pipeline.categorize import load_categories, load_rules
from pipeline.utils import CONFIG_DIR, ensure_data_files, init_db, list_data_files, load_overrides

st.set_page_config(page_title="Budget Sensei • Settings", layout="wide")


def _display_rules():
    rules = load_rules()
    if not rules:
        st.warning("No rules configured.")
        return
    rows = []
    for category, entries in rules.items():
        for entry in entries:
            rows.append({"category": category, "pattern": entry.get("pattern", "")})
    st.table(pd.DataFrame(rows))


def _display_categories():
    categories = load_categories()
    st.write(categories if categories else "No categories configured")


def _artifact_status():
    data_dir = CONFIG_DIR.parents[1] / "data"
    vectorizer = data_dir / "vectorizer.pkl"
    model = data_dir / "model.pkl"
    st.write(
        {
            "vectorizer_exists": vectorizer.exists(),
            "model_exists": model.exists(),
        }
    )


def _overrides_section():
    overrides = load_overrides()
    st.write(f"Total overrides: {len(overrides)}")
    if not overrides.empty:
        st.dataframe(overrides.tail(50))


def main() -> None:
    ensure_data_files()
    init_db()
    st.title("⚙️ Settings")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Categories")
        _display_categories()
    with col2:
        st.subheader("Rules")
        _display_rules()

    st.subheader("Model artifacts")
    _artifact_status()

    st.subheader("Overrides")
    _overrides_section()

    st.subheader("Data directory")
    files = list_data_files()
    if not files:
        st.info("Data directory is empty.")
    else:
        st.table(pd.DataFrame(files))


if __name__ == "__main__":
    main()
