from __future__ import annotations

import streamlit as st

from pipeline import categorize, normalize
from pipeline.utils import ensure_data_files, init_db, insert_transactions, load_transactions

st.set_page_config(page_title="Budget Sensei", layout="wide")


def _init_environment() -> None:
    ensure_data_files()
    init_db()


def main() -> None:
    _init_environment()
    st.title("💰 Budget Sensei")
    st.write("Import StatementSensei CSV exports, normalise them, categorise, and analyse your cash flow.")

    st.header("Upload bank statements")
    bank_name = st.text_input("Bank or account name", value="unknown")
    uploaded_files = st.file_uploader(
        "Upload one or more CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="Files exported from StatementSensei",
    )

    if uploaded_files:
        total_inserted = 0
        combined_frames = []
        for uploaded in uploaded_files:
            try:
                normalized = normalize.normalize_csv(uploaded, bank=bank_name or "unknown")
                categorized = categorize.categorize_transactions(normalized)
                inserted = insert_transactions(categorized)
                total_inserted += inserted
                combined_frames.append(categorized)
                st.success(f"Processed {len(categorized)} rows from {uploaded.name} (new: {inserted}).")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to process {uploaded.name}: {exc}")
        if total_inserted:
            st.toast(f"Inserted {total_inserted} new transactions")
        if combined_frames:
            preview = combined_frames[-1]
            st.subheader("Last imported preview")
            st.dataframe(preview.head(50))

    st.header("Recent transactions")
    transactions = load_transactions()
    if transactions.empty:
        st.info("No transactions stored yet. Upload a CSV to get started.")
    else:
        st.dataframe(transactions.head(100))


if __name__ == "__main__":
    main()
