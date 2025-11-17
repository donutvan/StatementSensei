from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st

from webapp.categorization import categorize_dataframe, record_overrides

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator


def render_metric(column: "DeltaGenerator", title: str, value: str, title_color: str = "#262730", value_color: str = "#262730") -> None:
    column.markdown(
        f"""
        <div style=\"text-align:center;\">
            <div style=\"font-size:16px; color:{title_color};\">{title}</div>
            <div style=\"font-size:36px; color:{value_color};\">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_stacked_bar_chart(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("Add statements to view monthly cash flow trends.")
        return

    income_trace = go.Bar(
        x=df.index,
        y=df["Income"],
        name="Income",
        marker={"color": "#00CEAA", "cornerradius": 6},
        hovertext=[f"${s:,.2f}" for s in df["Income"]],
        hoverinfo="text+name",
        offsetgroup=0,
    )
    expenses_trace = go.Bar(
        x=df.index,
        y=[-expense for expense in df["Expenses"]],
        name="Expenses",
        marker={"color": "#F63366", "cornerradius": 6},
        hovertext=[f"${s:,.2f}" for s in df["Expenses"]],
        hoverinfo="text+name",
        offsetgroup=0,
    )
    savings_trace = go.Scatter(
        x=df.index,
        y=df["Net"],
        name="Savings",
        mode="lines",
        line={"color": "black", "width": 3},
        hoverinfo="text+name",
        text=[f"${s:,.2f}" for s in df["Net"]],
    )
    layout = go.Layout(
        title="Monthly Cash Flow",
        xaxis={"title": "Month", "showgrid": False, "dtick": "M1"},
        yaxis={
            "title": "Amount",
            "showgrid": False,
            "zeroline": True,
            "zerolinecolor": "#EFEFEF",
            "zerolinewidth": 2,
            "tickformat": "$,.1s",
        },
        barmode="relative",
        hovermode="x unified",
        bargap=0.25,
        showlegend=True,
    )
    fig = go.Figure(data=[income_trace, expenses_trace, savings_trace], layout=layout)
    st.plotly_chart(fig, use_container_width=True)


def show_cumulative_spend(df: pd.DataFrame) -> None:
    if df.empty:
        return
    current_year = df["date"].dt.year.max()
    ytd = df[df["date"].dt.year == current_year].sort_values("date")
    ytd["cumulative_spend"] = ytd["debit"].cumsum()
    line = go.Scatter(
        x=ytd["date"],
        y=ytd["cumulative_spend"],
        name="YTD Spend",
        mode="lines+markers",
        line={"color": "#F63366"},
    )
    layout = go.Layout(title=f"{current_year} Cumulative Spending", yaxis={"title": "Amount ($)"})
    st.plotly_chart(go.Figure(data=[line], layout=layout), use_container_width=True)


def show_cashflow_volatility(monthly: pd.DataFrame) -> None:
    if monthly.empty:
        return
    volatility = monthly["Net"].rolling(3, min_periods=1).std().fillna(0)
    line = go.Scatter(
        x=monthly.index,
        y=volatility,
        name="Rolling 3M Net Std",
        line={"color": "#636EFA"},
    )
    layout = go.Layout(title="Cashflow Volatility", yaxis={"title": "Std Dev ($)"})
    st.plotly_chart(go.Figure(data=[line], layout=layout), use_container_width=True)


def show_top_merchants(df: pd.DataFrame) -> None:
    spenders = (
        df[df["debit"] > 0]
        .groupby(["description_clean", "category"], dropna=False)["debit"]
        .sum()
        .reset_index()
        .rename(columns={"description_clean": "Merchant", "debit": "Total Spend"})
        .sort_values("Total Spend", ascending=False)
        .head(10)
    )
    if spenders.empty:
        return
    st.dataframe(spenders, hide_index=True, use_container_width=True)


def show_category_deviation(df: pd.DataFrame) -> None:
    if df.empty:
        return
    monthly = (
        df.set_index("date")
        .groupby("category")
        .resample("MS")["debit"]
        .sum()
        .reset_index()
    )
    if monthly.empty:
        return
    monthly["month"] = monthly["date"].dt.to_period("M")
    last_month = monthly["month"].max()
    history = sorted(monthly["month"].unique())
    prior_months = [period for period in history if period < last_month][-3:]
    if not prior_months:
        return
    recent = monthly[monthly["month"] == last_month]
    baseline = monthly[monthly["month"].isin(prior_months)].groupby("category")["debit"].mean().reset_index()
    merged = recent.merge(baseline, on="category", how="left", suffixes=("_recent", "_baseline"))
    merged["delta_pct"] = ((merged["debit_recent"] - merged["debit_baseline"]) / merged["debit_baseline"].replace(0, np.nan)) * 100
    merged = merged.fillna({"debit_baseline": 0, "delta_pct": 0})
    merged = merged.rename(columns={
        "debit_recent": "Last Month",
        "debit_baseline": "3M Avg",
        "delta_pct": "% vs Avg",
    })
    merged["Last Month"] = merged["Last Month"].round(2)
    merged["3M Avg"] = merged["3M Avg"].round(2)
    merged["% vs Avg"] = merged["% vs Avg"].round(1)
    st.dataframe(merged.sort_values("% vs Avg", ascending=False), hide_index=True, use_container_width=True)


def detect_recurring_payments(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Merchant", "Category", "Months", "Avg Amount"])
    temp = df.copy()
    temp["month"] = temp["date"].dt.to_period("M")
    grouped = (
        temp.groupby(["description_fingerprint", "description_raw", "category"], dropna=False)
        .agg(
            months_seen=("month", lambda s: s.nunique()),
            avg_amount=("amount", "mean"),
            amount_std=("amount", "std"),
        )
        .reset_index()
    )
    recurring = grouped[(grouped["months_seen"] >= 3) & (grouped["amount_std"].fillna(0) < grouped["avg_amount"].abs() * 0.05 + 1)]
    recurring = recurring.sort_values(["months_seen", "avg_amount"], ascending=[False, True]).head(10)
    recurring = recurring.rename(
        columns={
            "description_raw": "Merchant",
            "category": "Category",
            "months_seen": "Months",
            "avg_amount": "Avg Amount",
        }
    )
    return recurring[["Merchant", "Category", "Months", "Avg Amount"]]


def spend_entropy(category_totals: pd.Series) -> float:
    positive = category_totals[category_totals > 0]
    total = positive.sum()
    if total <= 0 or positive.empty:
        return 0.0
    proportions = positive / total
    entropy = -(proportions * np.log(proportions)).sum()
    max_entropy = math.log(len(proportions)) if len(proportions) > 1 else 1
    if max_entropy == 0:
        return 0.0
    return round(float(entropy / max_entropy), 2)


st.markdown("# Financial Dashboard")

if "df" not in st.session_state:
    if st.button("Convert a bank statement"):
        st.switch_page("app.py")
else:
    df: pd.DataFrame = st.session_state["df"].copy()
    if df.empty:
        st.info("Upload a statement to explore the dashboard.")
        st.stop()

    df["date"] = pd.to_datetime(df["date"])

    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    banks = sorted(df["bank"].dropna().unique())
    accounts = sorted(df["account_number"].dropna().unique())
    currencies = sorted(df["currency"].dropna().unique())

    st.subheader("Filters")
    bank_col, account_col = st.columns(2)
    selected_banks = bank_col.multiselect("Banks", options=banks, default=banks)
    selected_accounts = account_col.multiselect(
        "Accounts",
        options=accounts,
        default=accounts,
        help="Filter specific account numbers when multiple statements are combined.",
    )

    date_col, currency_col = st.columns([2, 1])
    selected_range = date_col.date_input(
        "Statement period",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    currency_options = ["All"] + currencies
    selected_currency = currency_col.selectbox("Currency", options=currency_options)

    filtered_df = df.copy()
    if banks:
        if selected_banks:
            filtered_df = filtered_df[filtered_df["bank"].isin(selected_banks)]
        else:
            filtered_df = filtered_df.iloc[0:0]
    if accounts:
        if selected_accounts:
            filtered_df = filtered_df[filtered_df["account_number"].isin(selected_accounts)]
        else:
            filtered_df = filtered_df.iloc[0:0]
    if isinstance(selected_range, (tuple, list)):
        start_date, end_date = selected_range
    else:
        start_date, end_date = selected_range, selected_range
    filtered_df = filtered_df[(filtered_df["date"].dt.date >= start_date) & (filtered_df["date"].dt.date <= end_date)]
    if selected_currency != "All":
        filtered_df = filtered_df[filtered_df["currency"] == selected_currency]

    if filtered_df.empty:
        st.warning("No transactions match the selected filters.")
        st.stop()

    monthly = (
        filtered_df.sort_values("date")
        .set_index("date")
        .resample("MS")
        .agg({"credit": "sum", "debit": "sum", "amount": "sum"})
        .rename(columns={"credit": "Income", "debit": "Expenses", "amount": "Net"})
    )

    total_income = monthly["Income"].sum()
    total_expenses = monthly["Expenses"].sum()
    total_savings = monthly["Net"].sum()
    savings_rate = total_savings / total_income * 100 if total_income else 0
    median_spend = filtered_df.loc[filtered_df["debit"] > 0, "debit"].median()
    entropy_score = spend_entropy(filtered_df.groupby("category")["debit"].sum())
    monthly["savings_rate"] = monthly.apply(
        lambda row: (row["Net"] / row["Income"]) * 100 if row["Income"] else 0,
        axis=1,
    )
    savings_trend = monthly["savings_rate"].iloc[-1] - monthly["savings_rate"].iloc[-4:-1].mean() if len(monthly) > 1 else 0

    col1, col2, col3, col4 = st.columns(4)
    render_metric(col1, "Income", f"${total_income:,.0f}", value_color="#00CEAA")
    render_metric(col2, "Expenses", f"${total_expenses:,.0f}", value_color="#F63366")
    render_metric(col3, "Savings Rate", f"{savings_rate:.1f}%")
    trend_prefix = "+" if savings_trend >= 0 else ""
    render_metric(col4, "Savings Trend (vs 3M avg)", f"{trend_prefix}{savings_trend:.1f}pp")

    st.caption(
        f"Median spend: ${median_spend:,.0f} · Spend entropy: {entropy_score:.2f} (0=concentrated, 1=diversified)"
    )

    show_stacked_bar_chart(monthly)
    show_cumulative_spend(filtered_df)
    show_cashflow_volatility(monthly)

    st.subheader("Top Merchants & Subscriptions")
    show_top_merchants(filtered_df)

    st.subheader("Category deviation vs 3-month baseline")
    show_category_deviation(filtered_df)

    st.subheader("Recurring payments detected")
    recurring = detect_recurring_payments(filtered_df)
    if recurring.empty:
        st.info("No recurring debits detected for the selected window yet.")
    else:
        recurring["Avg Amount"] = recurring["Avg Amount"].round(2)
        st.dataframe(recurring, hide_index=True, use_container_width=True)

    st.subheader("Category overrides")
    editor_source = filtered_df.sort_values("date", ascending=False).head(200)[
        [
            "transaction_id",
            "date",
            "description_raw",
            "amount",
            "category",
            "categorization_source",
            "description_fingerprint",
        ]
    ].copy()
    editor_source = editor_source.rename(columns={
        "description_raw": "Description",
        "amount": "Amount",
        "category": "Category",
        "categorization_source": "Source",
    })
    edited = st.data_editor(
        editor_source,
        hide_index=True,
        use_container_width=True,
        column_config={
            "transaction_id": st.column_config.TextColumn("Transaction ID", disabled=True),
            "description_fingerprint": st.column_config.TextColumn("Fingerprint", disabled=True),
            "Amount": st.column_config.NumberColumn(format="$%.2f", disabled=True),
            "date": st.column_config.DateColumn("Date", disabled=True),
            "Source": st.column_config.TextColumn(disabled=True),
        },
        num_rows="fixed",
        key="category-editor",
    )

    if st.button("Save overrides", type="primary"):
        original_categories = editor_source.set_index("transaction_id")["Category"]
        fingerprints = editor_source.set_index("transaction_id")["description_fingerprint"]
        edited_categories = edited.set_index("transaction_id")["Category"]
        overrides: dict[str, str] = {}
        for tid, new_value in edited_categories.items():
            if tid not in original_categories.index:
                continue
            if new_value and new_value != original_categories.loc[tid]:
                fingerprint = fingerprints.loc[tid]
                overrides[fingerprint] = new_value
        if overrides:
            record_overrides(overrides)
            st.session_state["df"] = categorize_dataframe(st.session_state["df"])
            st.success(f"Saved {len(overrides)} override(s). Dashboard will refresh.")
            st.rerun()
        else:
            st.info("No category changes detected.")
