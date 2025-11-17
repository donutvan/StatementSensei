from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objs as go
import streamlit as st

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator


def render_metric(column: "DeltaGenerator", title, value, title_color="#262730", value_color="#262730"):
    column.markdown(
        f"""
        <div style="text-align:center;">
            <div style="font-size:16px; color:{title_color};">{title}</div>
            <div style="font-size:36px; color:{value_color};">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_stacked_bar_chart(df: pd.DataFrame):
    income_trace = go.Bar(
        x=df.index,
        y=df["Income"],
        name="Income",
        marker={"color": "#00CEAA", "cornerradius": 10},
        hovertext=[f"${s:,.2f}" for s in df["Income"]],
        hoverinfo="text+name",
        offsetgroup=0,
    )

    expenses_trace = go.Bar(
        x=df.index,
        y=[-expense for expense in df["Expenses"]],
        name="Expenses",
        marker={"color": "#F63366", "cornerradius": 10},
        hovertext=[f"${s:,.2f}" for s in df["Expenses"]],
        hoverinfo="text+name",
        offsetgroup=0,
    )

    savings_trace = go.Scatter(
        x=df.index,
        y=df["Net"],
        name="Savings",
        mode="lines",
        line={"color": "black", "width": 4},
        hoverinfo="text+name",
        text=[f"${s:,.2f}" for s in df["Net"]],
    )

    layout = go.Layout(
        title="Cash Flow",
        title_font={"size": 26},
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
        bargap=0.5,
        showlegend=False,
    )

    fig = go.Figure(data=[income_trace, expenses_trace, savings_trace], layout=layout)
    chart = st.plotly_chart(fig, use_container_width=True)

    total_income = round(df["Income"].sum())
    total_expenses = round(df["Expenses"].sum())
    total_savings = round(df["Net"].sum())

    # Avoid division by zero
    savings_rate = total_savings / total_income * 100 if total_income > 0 else 0
    formatted_savings_rate = f"{savings_rate:.2f}%"
    formatted_total_savings = f"${total_savings:,.0f}"
    formatted_total_savings = f"-${abs(total_savings):,}" if total_savings < 0 else f"${total_savings:,}"

    col1, col2, col3, col4 = st.columns(4)

    if chart:
        render_metric(col1, "Income", f"${total_income:,}", value_color="#00CEAA")
        render_metric(col2, "Expenses", f"${total_expenses:,}", value_color="#F63366")
        render_metric(col3, "Total Savings", formatted_total_savings)
        render_metric(col4, "Savings Rate", formatted_savings_rate)


st.markdown("# Visualizations")

if "df" not in st.session_state:
    switch_page_button = st.button("Convert a bank statement")
    if switch_page_button:
        st.switch_page("app.py")
else:
    df: pd.DataFrame = st.session_state["df"].copy()
    df["date"] = pd.to_datetime(df["date"])

    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    banks = sorted(df["bank"].dropna().unique())
    accounts = sorted(df["account_number"].dropna().unique())
    currencies = sorted(df["currency"].dropna().unique())

    st.subheader("Filters")
    bank_col, account_col = st.columns(2)
    selected_banks = bank_col.multiselect(
        "Banks",
        options=banks,
        default=banks,
    )
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
    filtered_df = filtered_df[
        (filtered_df["date"].dt.date >= start_date)
        & (filtered_df["date"].dt.date <= end_date)
    ]
    if selected_currency != "All":
        filtered_df = filtered_df[filtered_df["currency"] == selected_currency]

    if filtered_df.empty:
        st.warning("No transactions match the selected filters.")
    else:
        monthly = (
            filtered_df.sort_values("date")
            .set_index("date")
            .resample("MS")
            .agg({"credit": "sum", "debit": "sum", "amount": "sum"})
            .rename(columns={"credit": "Income", "debit": "Expenses", "amount": "Net"})
        )

        show_stacked_bar_chart(monthly)

        st.caption(
            f"Showing {len(filtered_df)} transactions from {start_date:%b %d, %Y} to "
            f"{end_date:%b %d, %Y}."
        )

        st.subheader("Bank & Account Summary")
        summary = (
            filtered_df.groupby(["bank", "account_number", "currency"], dropna=False)["amount"]
            .sum()
            .reset_index()
            .rename(columns={"amount": "net_total"})
        )
        summary["net_total"] = summary["net_total"].round(2)
        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True,
        )
