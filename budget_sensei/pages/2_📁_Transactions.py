from __future__ import annotations

import pandas as pd
import streamlit as st

from pipeline.categorize import load_categories
from pipeline.utils import ensure_data_files, init_db, load_transactions, save_override

st.set_page_config(page_title="Budget Sensei • Transactions", layout="wide")


def _filter_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    min_date, max_date = df["date"].min(), df["date"].max()
    col1, col2 = st.columns(2)
    with col1:
        date_range = st.date_input("Date range", value=(min_date.date(), max_date.date()))
    with col2:
        categories = load_categories()
        category_filter = st.multiselect("Categories", options=categories, default=categories)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df = df[(df["date"] >= start) & (df["date"] <= end)]
    if category_filter:
        df = df[df["category"].isin(category_filter)]
    search_text = st.text_input("Search description")
    if search_text:
        df = df[df["description"].str.contains(search_text, case=False, na=False)]
    return df


def _override_form(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No transactions available for overrides.")
        return
    df = df.sort_values("date", ascending=False)
    display_options = {
        f"{row['date']} • {row['description']} • {row['amount']:.2f}": row["transaction_hash"]
        for _, row in df.head(200).iterrows()
    }
    selection = st.selectbox("Choose a transaction", options=list(display_options.keys()))
    categories = load_categories()
    new_category = st.selectbox("Assign category", options=categories)
    if st.button("Save override"):
        save_override(display_options[selection], new_category)
        st.success("Override saved and applied.")


def main() -> None:
    ensure_data_files()
    init_db()
    st.title("📁 Transactions")
    transactions = load_transactions()
    if transactions.empty:
        st.info("No transactions stored yet. Upload data on the main page.")
        return
    filtered = _filter_transactions(transactions)
    st.dataframe(filtered.sort_values("date", ascending=False))

    st.subheader("Manual category overrides")
    _override_form(filtered)


if __name__ == "__main__":
    main()
