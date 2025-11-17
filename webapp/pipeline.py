"""Minimal ETL helpers to keep the dataframe schema predictable."""

from __future__ import annotations

import hashlib
import re
from typing import Iterable

import pandas as pd

from webapp.models import ProcessedFile

NORMALIZED_COLUMNS = [
    "transaction_id",
    "date",
    "description_raw",
    "description_clean",
    "description_fingerprint",
    "amount",
    "direction",
    "balance",
    "source_file",
    "account_type",
    "bank",
    "currency",
    "account_number",
    "statement_start",
    "statement_end",
    "credit",
    "debit",
    "is_credit",
    "is_debit",
]


def extract_transactions(processed_files: Iterable[ProcessedFile]) -> pd.DataFrame:
    """Transform ProcessedFile objects into a raw dataframe."""

    frames: list[pd.DataFrame] = []
    for file in processed_files:
        df = pd.DataFrame(file.transactions)
        if df.empty:
            continue
        df["source_file"] = file.metadata.source_file
        df["bank"] = file.metadata.bank_name
        df["currency"] = file.metadata.currency
        df["account_number"] = file.metadata.account_number
        df["account_type"] = file.metadata.account_type
        df["statement_start"] = file.metadata.statement_start
        df["statement_end"] = file.metadata.statement_end
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["date", "description", "amount"])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.rename(columns={"description": "description_raw"})
    if "description_raw" not in combined:
        combined["description_raw"] = ""
    return combined


def normalize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce the raw dataframe into a predictable schema."""

    if df.empty:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS + ["description"])

    normalized = df.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.date
    normalized["statement_start"] = pd.to_datetime(normalized["statement_start"], errors="coerce").dt.date
    normalized["statement_end"] = pd.to_datetime(normalized["statement_end"], errors="coerce").dt.date

    normalized["description_raw"] = normalized["description_raw"].fillna("").astype(str)
    normalized["description_clean"] = normalized["description_raw"].apply(_clean_description)
    normalized["description_fingerprint"] = normalized["description_clean"].apply(_fingerprint)

    normalized["amount"] = pd.to_numeric(normalized["amount"], errors="coerce").fillna(0.0).round(2)
    normalized["direction"] = normalized["amount"].apply(lambda value: "credit" if value >= 0 else "debit")

    if "balance" not in normalized.columns:
        normalized["balance"] = normalized.get("running_balance")
    normalized["balance"] = pd.to_numeric(normalized["balance"], errors="coerce")

    normalized["credit"] = normalized["amount"].clip(lower=0)
    normalized["debit"] = normalized["amount"].clip(upper=0).abs()
    normalized["is_credit"] = normalized["direction"] == "credit"
    normalized["is_debit"] = normalized["direction"] == "debit"

    normalized["description"] = normalized["description_raw"]
    normalized["transaction_id"] = normalized.apply(_build_transaction_id, axis=1)

    # Ensure all expected columns exist
    for column in NORMALIZED_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = None

    return normalized[NORMALIZED_COLUMNS + ["description"]]


def _clean_description(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9 ]+", " ", text or "").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.upper()


def _fingerprint(text: str) -> str:
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_transaction_id(row: pd.Series) -> str:
    raw = "|".join(
        [
            str(row.get("source_file", "")),
            str(row.get("account_number", "")),
            str(row.get("date", "")),
            str(row.get("amount", "")),
            row.get("description_raw", ""),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
