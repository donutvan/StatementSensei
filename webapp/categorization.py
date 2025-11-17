"""Hybrid categorisation pipeline mixing heuristics and lightweight NLP."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from webapp.storage import load_overrides, upsert_overrides

TRAINING_DATA_PATH = Path(__file__).with_name("data") / "default_training_samples.json"

RULES: list[tuple[str, str, str]] = [
    ("groceries", "Groceries", r"FAIRPRICE|NTUC|SHENG SIONG|COLD STORAGE|GIANT"),
    ("food_delivery", "Food Delivery", r"GRABFOOD|FOODPANDA|DELIVEROO"),
    ("transport", "Transport", r"GRAB\b|GOJEK|COMFORT|UBER|MRT|EZ[ -]?LINK"),
    ("subscriptions", "Subscriptions", r"NETFLIX|SPOTIFY|YOUTUBE|DISNEY|MICROSOFT|APPLE MUSIC"),
    ("utilities", "Utilities", r"SP GROUP|SINGTEL|STARHUB|CIRCULAR WATER|PUB"),
    ("salary", "Salary", r"PAYROLL|SALARY|GIRO CREDIT|PAYMENT FROM"),
    ("investments", "Investments", r"TIGER\s?BROKERS|IBKR|ROBINHOOD|COINBASE"),
    ("fees", "Fees", r"FEE|CHARGE|INTEREST"),
]


@dataclass
class CategoryResult:
    category: str
    source: str
    confidence: float
    rule_used: str | None = None

    def as_tuple(self) -> tuple[str, str, float, str | None]:
        return self.category, self.source, self.confidence, self.rule_used


class RuleBasedCategorizer:
    def __init__(self, rules: Iterable[tuple[str, str, str]]):
        self._rules = [
            (name, category, re.compile(pattern, flags=re.IGNORECASE))
            for name, category, pattern in rules
        ]

    def predict(self, text: str) -> CategoryResult | None:
        for name, category, pattern in self._rules:
            if pattern.search(text):
                return CategoryResult(category, "regex", 0.98, name)
        return None


class SimpleNaiveBayesClassifier:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._token_totals: dict[str, Counter[str]] = defaultdict(Counter)
        self._category_totals: Counter[str] = Counter()
        self._vocab: set[str] = set()
        self._trained = False

    def fit(self, samples: Iterable[tuple[str, str]]) -> None:
        for description, category in samples:
            tokens = _tokenize(description)
            if not tokens:
                continue
            self._category_totals[category] += 1
            for token in tokens:
                self._token_totals[category][token] += 1
                self._vocab.add(token)
        self._trained = bool(self._category_totals)

    def predict(self, text: str) -> CategoryResult | None:
        if not self._trained:
            return None
        tokens = _tokenize(text)
        if not tokens:
            return None

        vocab_size = max(len(self._vocab), 1)
        log_scores: dict[str, float] = {}
        total_samples = sum(self._category_totals.values())

        for category, category_total in self._category_totals.items():
            log_prob = math.log(category_total / total_samples)
            token_total = sum(self._token_totals[category].values())
            for token in tokens:
                token_freq = self._token_totals[category][token]
                log_prob += math.log((token_freq + self.alpha) / (token_total + self.alpha * vocab_size))
            log_scores[category] = log_prob

        if not log_scores:
            return None

        max_category = max(log_scores, key=log_scores.get)
        # Convert log scores into pseudo confidence
        exp_scores = {cat: math.exp(score - log_scores[max_category]) for cat, score in log_scores.items()}
        total = sum(exp_scores.values())
        confidence = exp_scores[max_category] / total if total else 0.5
        return CategoryResult(max_category, "ml_model", round(confidence, 2), None)


class HybridCategorizer:
    def __init__(self):
        self.rule_engine = RuleBasedCategorizer(RULES)
        self.ml_engine = SimpleNaiveBayesClassifier()
        self._load_training_data()

    def _load_training_data(self) -> None:
        samples = []
        if TRAINING_DATA_PATH.exists():
            try:
                payload = json.loads(TRAINING_DATA_PATH.read_text())
            except json.JSONDecodeError:
                payload = []
            for item in payload:
                samples.append((item.get("description", ""), item.get("category", "Uncategorized")))
        self.ml_engine.fit(samples)

    def assign_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            df["category"] = []
            df["categorization_source"] = []
            df["categorization_confidence"] = []
            df["categorization_rule"] = []
            return df

        overrides = load_overrides()
        assignments = df.apply(lambda row: self._classify_row(row, overrides), axis=1, result_type="expand")
        assignments.columns = ["category", "categorization_source", "categorization_confidence", "categorization_rule"]
        enriched = df.copy()
        for column in assignments.columns:
            enriched[column] = assignments[column]
        return enriched

    def _classify_row(self, row: pd.Series, overrides: dict[str, str]) -> tuple[str, str, float, str | None]:
        fingerprint = row.get("description_fingerprint", "")
        description = row.get("description_clean") or row.get("description_raw", "")

        if fingerprint and overrides.get(fingerprint):
            return overrides[fingerprint], "user_override", 1.0, None

        rule_result = self.rule_engine.predict(description)
        if rule_result:
            return rule_result.as_tuple()

        ml_result = self.ml_engine.predict(description)
        if ml_result:
            return ml_result.as_tuple()

        return "Uncategorized", "fallback", 0.0, None


def categorize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Public helper consumed by the Streamlit app."""

    categorizer = HybridCategorizer()
    return categorizer.assign_categories(df)


def record_overrides(updates: dict[str, str]) -> dict[str, str]:
    """Expose persistence so UI components can store new labels."""

    return upsert_overrides(updates)


def _tokenize(text: str) -> list[str]:
    cleaned = re.sub(r"[^A-Za-z0-9 ]+", " ", text or "").strip().lower()
    return [token for token in cleaned.split() if token]
