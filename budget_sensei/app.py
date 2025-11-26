from __future__ import annotations

import streamlit as st

from budget_sensei.parsers.csv_parser import normalize_csv
from budget_sensei.parsers.pdf_parser import PdfImportError, pdf_to_dataframe
from budget_sensei.services.categorization import categorize_transactions
from budget_sensei.storage import ensure_data_files, init_db, insert_transactions, load_transactions

st.set_page_config(page_title="Budget Sensei", layout="wide")


def _init_environment() -> None:
    ensure_data_files()
    init_db()


def _process_file(uploaded, bank_name: str, pdf_password: str | None) -> tuple[int, int]:
    if str(uploaded.name).lower().endswith(".pdf") or getattr(uploaded, "type", "").endswith("pdf"):
        normalized = pdf_to_dataframe(uploaded, password=pdf_password or None)
    else:
        normalized = normalize_csv(uploaded, bank=bank_name or "unknown")
    categorized = categorize_transactions(normalized)
    inserted = insert_transactions(categorized)
    return inserted, len(categorized)


def main() -> None:
    _init_environment()
    st.title("Budget Sensei")
    st.write(
        "Import StatementSensei PDFs or CSV exports, normalise them, categorise, and analyse your cash flow."
    )

    st.header("Upload bank statements")
    bank_name = st.text_input("Bank or account name", value="unknown")
    pdf_password = st.text_input("PDF password (optional)", type="password")
    uploaded_files = st.file_uploader(
        "Upload one or more statement files", type=["csv", "pdf"], accept_multiple_files=True
    )

    if uploaded_files:
        total_inserted = 0
        total_rows = 0
        for uploaded in uploaded_files:
            try:
                inserted, count = _process_file(uploaded, bank_name, pdf_password)
                total_inserted += inserted
                total_rows += count
                st.success(f"Processed {count} rows from {uploaded.name} (new: {inserted}).")
            except PdfImportError as exc:
                st.error(f"PDF processing failed for {uploaded.name}: {exc}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to process {uploaded.name}: {exc}")
        if total_inserted:
            st.toast(f"Inserted {total_inserted} new transactions")
        if total_rows:
            st.info(f"Processed a total of {total_rows} rows across {len(uploaded_files)} file(s).")

    st.header("Recent transactions")
    transactions = load_transactions()
    if transactions.empty:
        st.info("No transactions stored yet. Upload a CSV or PDF to get started.")
    else:
        st.dataframe(transactions.head(100))


if __name__ == "__main__":
    main()
