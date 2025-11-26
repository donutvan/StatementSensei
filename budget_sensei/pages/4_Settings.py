from __future__ import annotations

from pathlib import Path

import streamlit as st

from budget_sensei.rules import add_category, load_categories
from budget_sensei.storage import ensure_data_files, get_data_dir, list_data_files

st.set_page_config(page_title="Budget Sensei • Settings", layout="wide")

RULES_PATH = Path(__file__).resolve().parents[1] / "config" / "rules.yaml"


def _render_rules():
    st.subheader("Rules file")
    if RULES_PATH.exists():
        with open(RULES_PATH, "r", encoding="utf-8") as file:
            st.code(file.read(), language="yaml")
    else:
        st.info("rules.yaml not found")


def _render_categories():
    st.subheader("Categories")
    categories = load_categories()
    st.write(categories)
    new_cat = st.text_input("Add category")
    if st.button("Save category"):
        updated = add_category(new_cat)
        st.success(f"Saved. Total categories: {len(updated)}")


def _render_ml_artifacts():
    st.subheader("Machine learning artifacts")
    data_dir = get_data_dir()
    vectorizer = data_dir / "vectorizer.pkl"
    model = data_dir / "model.pkl"
    status = {"vectorizer": vectorizer.exists(), "model": model.exists()}
    st.json(status)


def _render_data_files():
    st.subheader("Data directory contents")
    files = list_data_files()
    if not files:
        st.info("No data files yet.")
        return
    st.table(files)


def main() -> None:
    ensure_data_files()
    st.title("Settings")
    _render_rules()
    _render_categories()
    _render_ml_artifacts()
    _render_data_files()


if __name__ == "__main__":
    main()
