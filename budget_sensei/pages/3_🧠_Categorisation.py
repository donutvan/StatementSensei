from __future__ import annotations

import streamlit as st

from pipeline.categorize import load_categories
from pipeline.train_model import train_model
from pipeline.utils import append_labeled_samples, ensure_data_files, init_db, load_labeled_samples, load_transactions

st.set_page_config(page_title="Budget Sensei • Categorisation", layout="wide")


def _uncategorised_transactions(df):
    if df.empty:
        return df
    return df[df["category"].isin(["", "uncategorised", None])]


def _label_form(df) -> None:
    if df.empty:
        st.info("No uncategorised transactions available.")
        return
    sample = df.sample(min(len(df), 200), random_state=42)
    options = {
        f"{row['date']} • {row['description']} • {row['amount']:.2f}": row
        for _, row in sample.iterrows()
    }
    selection = st.selectbox("Pick a transaction to label", list(options.keys()))
    categories = load_categories()
    choice = st.selectbox("Assign category", options=categories)
    if st.button("Save label"):
        row = options[selection]
        append_labeled_samples(
            [
                {
                    "transaction_hash": row["transaction_hash"],
                    "description": row["description"],
                    "amount": row["amount"],
                    "category": choice,
                }
            ]
        )
        st.success("Label saved. Train the model to include it.")


def _train_section():
    if st.button("Train classification model"):
        try:
            result = train_model()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Training failed: {exc}")
            return
        st.success(
            f"Training complete on {result['samples']} samples. Artifacts saved to data directory."
        )
        st.json(result["report"])


def main() -> None:
    ensure_data_files()
    init_db()
    st.title("🧠 Categorisation trainer")
    transactions = load_transactions()
    uncat = _uncategorised_transactions(transactions)
    st.metric("Uncategorised transactions", len(uncat))

    st.subheader("Label data")
    _label_form(uncat)

    labeled = load_labeled_samples()
    st.subheader("Labelled dataset overview")
    st.write(f"Total labelled samples: {len(labeled)}")
    if not labeled.empty:
        st.dataframe(labeled.tail(50))

    st.subheader("Model training")
    _train_section()


if __name__ == "__main__":
    main()
