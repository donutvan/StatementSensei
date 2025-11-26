from __future__ import annotations

import random

import pandas as pd
import streamlit as st

from budget_sensei.ml.training import train_model
from budget_sensei.rules import add_category, load_categories
from budget_sensei.services.categorization import categorize_transactions
from budget_sensei.storage import (
    append_labeled_samples,
    ensure_data_files,
    init_db,
    load_labeled_samples,
    load_transactions,
)

st.set_page_config(page_title="Budget Sensei • Categorisation", layout="wide")


def _sample_uncategorised(df: pd.DataFrame, sample_size: int = 20) -> pd.DataFrame:
    candidates = df[df["category"].isin(["", "uncategorised", None])]
    if candidates.empty:
        return pd.DataFrame()
    sample_size = min(sample_size, len(candidates))
    sampled_hashes = random.sample(list(candidates["transaction_hash"]), sample_size)
    return candidates[candidates["transaction_hash"].isin(sampled_hashes)]


def _label_transactions(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No uncategorised transactions to label.")
        return
    categories = load_categories()
    records = []
    for _, row in df.iterrows():
        st.write(f"{row['date']} • {row['description']} • {row['amount']:.2f}")
        selected = st.selectbox("Assign category", options=categories, key=row["transaction_hash"])
        records.append(
            {
                "transaction_hash": row["transaction_hash"],
                "description": row["description"],
                "amount": row["amount"],
                "category": selected,
            }
        )
    if st.button("Save labels"):
        append_labeled_samples(records)
        st.success("Labels saved.")


def _train_model_section() -> None:
    st.subheader("Train machine learning model")
    samples = load_labeled_samples()
    st.write(f"Labeled samples available: {len(samples)}")
    if st.button("Train model"):
        try:
            report = train_model()
            st.success(f"Model trained on {report['samples']} samples.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Training failed: {exc}")


def _apply_categorisation_controls(transactions: pd.DataFrame) -> None:
    st.subheader("Apply categorisation")
    if st.button("Apply rules/ML to uncategorised"):
        updated = categorize_transactions(transactions.copy())
        st.dataframe(updated.head(20))
        st.info("Results shown for review; persist by re-uploading or using reapply on Transactions page.")


def main() -> None:
    ensure_data_files()
    init_db()
    st.title("Categorisation")
    transactions = load_transactions()
    if transactions.empty:
        st.info("No transactions loaded yet.")
        return

    st.subheader("Label uncategorised transactions")
    samples = _sample_uncategorised(transactions)
    _label_transactions(samples)

    _train_model_section()
    _apply_categorisation_controls(transactions)

    st.subheader("Add new category")
    new_category = st.text_input("New category name")
    if st.button("Add category"):
        categories = add_category(new_category)
        st.success(f"Added. Total categories: {len(categories)}")


if __name__ == "__main__":
    main()
