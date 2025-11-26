from __future__ import annotations

import io
from typing import Optional

import pandas as pd
from dateutil import parser

from budget_sensei.storage import hash_transaction

DATE_CANDIDATES = [
    "date",
    "transaction_date",
    "posted",
    "post_date",
    "timestamp",
]

DESCRIPTION_CANDIDATES = [
    "description",
    "memo",
    "details",
    "narrative",
    "merchant",
]

AMOUNT_CANDIDATES = [
    "amount",
    "value",
    "transaction_amount",
]

CREDIT_CANDIDATES = ["credit", "deposit"]
DEBIT_CANDIDATES = ["debit", "withdrawal", "spend"]
DIRECTION_CANDIDATES = ["type", "direction", "dr_cr", "transaction_type"]


class NormalizationError(Exception):
    """Raised when a CSV file cannot be normalized."""


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]
    return df


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _parse_date_series(series: pd.Series) -> pd.Series:
    def _parse(value):
        if pd.isna(value):
            return pd.NaT
        try:
            return parser.parse(str(value)).date()
        except Exception as exc:  # noqa: BLE001
            raise NormalizationError(f"Unable to parse date value: {value}") from exc

    return series.apply(_parse)


def _compute_amount(df: pd.DataFrame) -> pd.Series:
    amount_col = _pick_column(df, AMOUNT_CANDIDATES)
    if amount_col:
        series = pd.to_numeric(df[amount_col], errors="coerce").fillna(0)
        return series
    credit_col = _pick_column(df, CREDIT_CANDIDATES)
    debit_col = _pick_column(df, DEBIT_CANDIDATES)
    if credit_col or debit_col:
        credit = pd.to_numeric(df.get(credit_col, 0), errors="coerce").fillna(0)
        debit = pd.to_numeric(df.get(debit_col, 0), errors="coerce").fillna(0)
        return credit - debit
    raise NormalizationError("No amount-like column found")


def _compute_direction(df: pd.DataFrame, amount: pd.Series) -> pd.Series:
    dir_col = _pick_column(df, DIRECTION_CANDIDATES)
    if dir_col:
        values = df[dir_col].astype(str).str.lower()
        direction = values.map(
            lambda x: "incoming" if any(token in x for token in ["credit", "cr", "deposit", "in"]) else "outgoing"
        )
        return direction
    return amount.apply(lambda x: "incoming" if x >= 0 else "outgoing")


def normalize_dataframe(df: pd.DataFrame, bank: str = "unknown", source_file: str = "uploaded") -> pd.DataFrame:
    if df.empty:
        raise NormalizationError("Uploaded file is empty")
    df = _standardize_columns(df)
    date_col = _pick_column(df, DATE_CANDIDATES)
    desc_col = _pick_column(df, DESCRIPTION_CANDIDATES)
    if not date_col or not desc_col:
        raise NormalizationError("Missing date or description columns")
    amounts = _compute_amount(df)
    directions = _compute_direction(df, amounts)
    parsed_dates = _parse_date_series(df[date_col])
    normalized = pd.DataFrame(
        {
            "date": parsed_dates,
            "description": df[desc_col].astype(str).str.strip(),
            "amount": amounts.astype(float),
            "direction": directions,
            "bank": bank or "unknown",
            "source_file": source_file,
        }
    )
    normalized.dropna(subset=["date", "description"], inplace=True)
    normalized["amount"] = normalized["amount"].fillna(0)
    normalized["category"] = "uncategorised"
    normalized["transaction_hash"] = normalized.apply(
        lambda row: hash_transaction(row["date"], row["description"], row["amount"]),
        axis=1,
    )
    return normalized


def normalize_csv(file_obj: io.BytesIO, bank: str = "unknown") -> pd.DataFrame:
    if hasattr(file_obj, "name"):
        source_name = getattr(file_obj, "name")
    else:
        source_name = "uploaded"
    df = pd.read_csv(file_obj)
    return normalize_dataframe(df, bank=bank, source_file=source_name)


__all__ = ["normalize_csv", "normalize_dataframe", "NormalizationError"]
