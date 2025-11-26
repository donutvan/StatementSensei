from __future__ import annotations

import pandas as pd


def detect_recurring_payments(transactions: pd.DataFrame, min_occurrences: int = 3) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame(columns=["description", "avg_interval_days", "occurrences", "avg_amount", "last_date"])
    df = transactions.copy()
    df["date"] = pd.to_datetime(df["date"])
    groups = []
    for description, group in df.groupby("description"):
        group = group.sort_values("date")
        if len(group) < min_occurrences:
            continue
        deltas = group["date"].diff().dt.days.dropna()
        if deltas.empty:
            continue
        avg_interval = deltas.mean()
        std_interval = deltas.std()
        if avg_interval <= 45 and (pd.isna(std_interval) or std_interval <= avg_interval * 0.5):
            groups.append(
                {
                    "description": description,
                    "avg_interval_days": round(avg_interval, 1),
                    "occurrences": len(group),
                    "avg_amount": round(group["amount"].mean(), 2),
                    "last_date": group["date"].max().date(),
                }
            )
    if not groups:
        return pd.DataFrame(columns=["description", "avg_interval_days", "occurrences", "avg_amount", "last_date"])
    return pd.DataFrame(groups).sort_values("occurrences", ascending=False)
