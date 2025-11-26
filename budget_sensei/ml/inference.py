from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import joblib
import pandas as pd

from budget_sensei.storage import get_data_dir


@lru_cache(maxsize=1)
def _load_model() -> Tuple[object | None, object | None]:
    data_dir = get_data_dir()
    vectorizer_path = data_dir / "vectorizer.pkl"
    model_path = data_dir / "model.pkl"
    if not vectorizer_path.exists() or not model_path.exists():
        return None, None
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    return vectorizer, model


def clear_model_cache() -> None:
    _load_model.cache_clear()


def predict_categories(df: pd.DataFrame) -> pd.DataFrame:
    vectorizer, model = _load_model()
    if vectorizer is None or model is None:
        return df
    mask = df["category"].isin(["", "uncategorised", None])
    if not mask.any():
        return df
    text = (df.loc[mask, "description"].fillna("") + " " + df.loc[mask, "amount"].abs().astype(str)).values
    features = vectorizer.transform(text)
    predictions = model.predict(features)
    df.loc[mask, "category"] = predictions
    return df


__all__ = ["predict_categories", "clear_model_cache"]
