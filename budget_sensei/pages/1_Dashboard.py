from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from budget_sensei.pipeline.recurring import detect_recurring_payments
from budget_sensei.storage import ensure_data_files, init_db, load_transactions

st.set_page_config(page_title="Budget Sensei • Dashboard", layout="wide")


def _prepare_transactions() -> pd.DataFrame:
    transactions = load_transactions()
    if transactions.empty:
        return transactions
    transactions["date"] = pd.to_datetime(transactions["date"])
    transactions["month"] = transactions["date"].dt.to_period("M").dt.to_timestamp()
    return transactions


def _monthly_cashflow_chart(df: pd.DataFrame):
    monthly = df.groupby("month", as_index=False)["amount"].sum()
    chart = (
        alt.Chart(monthly)
        .mark_bar()
        .encode(x="month:T", y="amount:Q", tooltip=["month", "amount"])
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)


def _category_spend_chart(df: pd.DataFrame):
    expenses = df[df["amount"] < 0]
    if expenses.empty:
        st.info("No expenses available for category chart.")
        return
    category_totals = (
        expenses.assign(spend=lambda x: x["amount"].abs())
        .groupby("category", as_index=False)["spend"].sum()
        .sort_values("spend", ascending=False)
    )
    chart = (
        alt.Chart(category_totals)
        .mark_bar()
        .encode(x="spend:Q", y=alt.Y("category:N", sort="-x"), tooltip=["category", "spend"])
    )
    st.altair_chart(chart, use_container_width=True)


def _top_merchants(df: pd.DataFrame):
    expenses = df[df["amount"] < 0]
    if expenses.empty:
        st.info("No merchant data yet.")
        return
    merchants = (
        expenses.assign(spend=lambda x: x["amount"].abs())
        .groupby("description", as_index=False)["spend"].sum()
        .sort_values("spend", ascending=False)
        .head(10)
    )
    st.table(merchants)


def _recurring_section(df: pd.DataFrame):
    recurring = detect_recurring_payments(df)
    if recurring.empty:
        st.info("No recurring payments detected yet.")
    else:
        st.dataframe(recurring)


def main() -> None:
    ensure_data_files()
    init_db()
    st.title("Dashboard")
    transactions = _prepare_transactions()
    if transactions.empty:
        st.info("Import transactions first via the main page.")
        return
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Monthly cash flow")
        _monthly_cashflow_chart(transactions)
    with col2:
        st.subheader("Category spend")
        _category_spend_chart(transactions)

    st.subheader("Top merchants")
    _top_merchants(transactions)

    st.subheader("Recurring payments")
    _recurring_section(transactions)


if __name__ == "__main__":
    main()
