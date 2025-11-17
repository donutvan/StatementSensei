from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from .utils import get_data_dir, load_labeled_samples


def _prepare_features(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    texts = df["description"].fillna("") + " " + df["amount"].abs().astype(str)
    labels = df["category"].astype(str)
    return texts, labels


def train_model(test_size: float = 0.2, random_state: int = 42) -> dict:
    samples = load_labeled_samples()
    samples.dropna(subset=["description", "category"], inplace=True)
    if samples.empty or samples["category"].nunique() < 2:
        raise ValueError("Need at least two categories of labeled samples to train")
    texts, labels = _prepare_features(samples)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    data_dir = get_data_dir()
    joblib.dump(vectorizer, data_dir / "vectorizer.pkl")
    joblib.dump(model, data_dir / "model.pkl")
    return {
        "samples": len(samples),
        "vectorizer_path": str(data_dir / "vectorizer.pkl"),
        "model_path": str(data_dir / "model.pkl"),
        "report": report,
    }
