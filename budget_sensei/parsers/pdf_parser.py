from __future__ import annotations

import pandas as pd
from monopoly.pdf import MissingPasswordError, PdfDocument

from webapp.helpers import parse_bank_statement

from budget_sensei.parsers.csv_parser import normalize_dataframe


class PdfImportError(Exception):
    """Raised when a PDF bank statement cannot be processed."""


def _unlock_document(document: PdfDocument, password: str | None = None) -> PdfDocument:
    if not document.is_encrypted:  # type: ignore[attr-defined]
        return document
    try:
        return document.unlock_document()
    except MissingPasswordError:
        pass

    if password:
        document.authenticate(password)
        if not document.is_encrypted:  # type: ignore[attr-defined]
            return document
    raise PdfImportError("Unable to unlock encrypted PDF; provide a valid password.")


def pdf_to_dataframe(uploaded_file, password: str | None = None) -> pd.DataFrame:
    file_bytes = uploaded_file.getvalue()
    document = PdfDocument(file_bytes=file_bytes)
    document._name = getattr(uploaded_file, "name", "uploaded.pdf")

    document = _unlock_document(document, password=password)

    processed = parse_bank_statement(document, password=password)
    df = pd.DataFrame(processed)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["bank"] = processed.metadata.bank_name
    df = df.drop(columns=["polarity"], errors="ignore")
    normalized = normalize_dataframe(df, bank=processed.metadata.bank_name, source_file=document.name)
    return normalized


__all__ = ["PdfImportError", "pdf_to_dataframe"]
