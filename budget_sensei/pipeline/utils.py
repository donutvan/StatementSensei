from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import duckdb
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def get_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def get_db_path() -> Path:
    return get_data_dir() / "transactions.db"


def ensure_data_files() -> None:
    data_dir = get_data_dir()
    labeled_path = data_dir / "labeled_samples.csv"
    overrides_path = data_dir / "user_overrides.csv"
    if not labeled_path.exists():
        pd.DataFrame(columns=[
            "transaction_hash",
            "description",
            "amount",
            "category",
        ]).to_csv(labeled_path, index=False)
    if not overrides_path.exists():
        pd.DataFrame(columns=["transaction_hash", "category"]).to_csv(overrides_path, index=False)


def connect_db(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    ensure_data_files()
    db_path = get_db_path()
    conn = duckdb.connect(str(db_path), read_only=read_only)
    return conn


def init_db() -> None:
    conn = connect_db()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_hash TEXT PRIMARY KEY,
            date DATE,
            description TEXT,
            amount DOUBLE,
            direction TEXT,
            bank TEXT,
            category TEXT,
            source_file TEXT,
            created_at TIMESTAMP
        )
        """
    )
    conn.close()


def insert_transactions(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    required_cols = {"transaction_hash", "date", "description", "amount", "direction", "bank", "category", "source_file"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    conn = connect_db()
    existing = conn.execute("SELECT transaction_hash FROM transactions").fetchdf()
    existing_hashes = set(existing["transaction_hash"]) if not existing.empty else set()
    new_df = df[~df["transaction_hash"].isin(existing_hashes)].copy()
    if new_df.empty:
        conn.close()
        return 0
    new_df["created_at"] = datetime.utcnow()
    conn.register("new_transactions", new_df)
    conn.execute(
        """
        INSERT INTO transactions
        SELECT transaction_hash, date, description, amount, direction, bank, category, source_file, created_at
        FROM new_transactions
        """
    )
    conn.close()
    return len(new_df)


def load_transactions() -> pd.DataFrame:
    conn = connect_db(read_only=True)
    df = conn.execute("SELECT * FROM transactions ORDER BY date DESC").fetchdf()
    conn.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def update_transaction_category(transaction_hash: str, category: str) -> None:
    conn = connect_db()
    conn.execute(
        "UPDATE transactions SET category = ? WHERE transaction_hash = ?",
        [category, transaction_hash],
    )
    conn.close()


def load_overrides() -> pd.DataFrame:
    ensure_data_files()
    path = get_data_dir() / "user_overrides.csv"
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=["transaction_hash", "category"])
    return df


def save_override(transaction_hash: str, category: str) -> None:
    overrides = load_overrides()
    if transaction_hash in set(overrides["transaction_hash"]):
        overrides.loc[overrides["transaction_hash"] == transaction_hash, "category"] = category
    else:
        overrides = pd.concat(
            [
                overrides,
                pd.DataFrame(
                    {"transaction_hash": [transaction_hash], "category": [category]}
                ),
            ],
            ignore_index=True,
        )
    overrides.to_csv(get_data_dir() / "user_overrides.csv", index=False)
    update_transaction_category(transaction_hash, category)


def list_data_files() -> List[dict]:
    ensure_data_files()
    data_dir = get_data_dir()
    files = []
    for path in sorted(data_dir.glob("*")):
        if path.is_file():
            files.append({
                "name": path.name,
                "size": path.stat().st_size,
                "modified": datetime.fromtimestamp(path.stat().st_mtime),
            })
    return files


def load_labeled_samples() -> pd.DataFrame:
    ensure_data_files()
    path = get_data_dir() / "labeled_samples.csv"
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=["transaction_hash", "description", "amount", "category"])
    return df


def append_labeled_samples(rows: Iterable[dict]) -> None:
    samples = load_labeled_samples()
    rows_list = list(rows)
    if not rows_list:
        return
    new_df = pd.DataFrame(rows_list)
    combined = pd.concat([samples, new_df], ignore_index=True)
    combined.drop_duplicates(subset=["transaction_hash"], keep="last", inplace=True)
    combined.to_csv(get_data_dir() / "labeled_samples.csv", index=False)


def hash_transaction(date_value, description: str, amount: float) -> str:
    base = f"{date_value}|{description}|{amount:.2f}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()
