# pylint: disable=unsubscriptable-object

from datetime import date, datetime
from typing import Any

import pandas as pd
import streamlit as st
from monopoly.banks import BankDetector, banks
from monopoly.generic import GenericBank
from monopoly.pdf import MissingOCRError, PdfDocument, PdfParser
from monopoly.pipeline import Pipeline
from monopoly.statements.base import SafetyCheckError
from pydantic import SecretStr

from webapp.models import ProcessedFile, TransactionMetadata


def _coerce_date(value: Any) -> date | None:
    if value is None:
        return None

    if isinstance(value, pd.Timestamp):
        return value.date()

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    return None


def _extract_statement_period(statement: Any) -> tuple[date | None, date | None]:
    period = getattr(statement, "statement_period", None) or getattr(statement, "period", None)
    if period is None:
        return None, None

    start_candidates = [
        "start",
        "start_date",
        "from_date",
        "from_dt",
        "from_date_inclusive",
    ]
    end_candidates = [
        "end",
        "end_date",
        "to_date",
        "to_dt",
        "through_date",
    ]

    start_date = None
    for attr in start_candidates:
        start_date = _coerce_date(getattr(period, attr, None))
        if start_date:
            break

    end_date = None
    for attr in end_candidates:
        end_date = _coerce_date(getattr(period, attr, None))
        if end_date:
            break

    return start_date, end_date


def _extract_metadata(statement: Any, bank_name: str) -> TransactionMetadata:
    account_number = getattr(statement, "account_number", None) or getattr(
        getattr(statement, "account", None), "number", None
    )
    currency = (
        getattr(statement, "currency", None)
        or getattr(getattr(statement, "config", None), "currency", None)
    )
    statement_start, statement_end = _extract_statement_period(statement)

    return TransactionMetadata(
        bank_name=bank_name,
        account_number=account_number,
        currency=currency,
        statement_start=statement_start,
        statement_end=statement_end,
    )


def build_pipeline(
    document: PdfDocument, password: str | None = None
) -> tuple[Pipeline, PdfParser]:
    analyzer = BankDetector(document)
    bank = analyzer.detect_bank(banks) or GenericBank
    parser = PdfParser(bank, document)
    passwords = [SecretStr(password)] if password else None
    pipeline = Pipeline(parser, passwords=passwords)
    return pipeline, parser


def parse_bank_statement(document: PdfDocument, password: str | None = None) -> ProcessedFile:
    try:
        pipeline, parser = build_pipeline(document, password)
    except MissingOCRError:
        st.info(f"No text found - {document.name}. Attempting to apply OCR.")
        with st.spinner(f"Adding OCR layer for {document.name}"):
            analyzer = BankDetector(document)
            bank = analyzer.detect_bank(banks) or GenericBank
            # certain PDFs have strange formats that can break the OCR,
            # so they need to be cropped before further processing
            if cropbox := bank.pdf_config.page_bbox:
                for page in document:
                    page.set_cropbox(cropbox)

            document = PdfParser.apply_ocr(document)
            pipeline, parser = build_pipeline(document, password)

    # skip initial safety check, and handle it outside the pipeline
    # so that we can raise a warning and still show transactions
    statement = pipeline.extract(safety_check=False)
    bank_name = parser.bank.__name__

    if statement.config.safety_check:
        try:
            statement.perform_safety_check()
        except SafetyCheckError:
            st.error(
                f"Safety check failed for {document.name}, transactions are incorrect or missing",
                icon="❗",
            )
        else:
            st.success(
                f"Safety check passed for {document.name}.",
                icon="✅",
            )
    if not statement.config.safety_check:
        st.warning(
            f"{bank_name} {statement.config.statement_type} statements have no safety check, "
            "please review your transactions and proceed with caution",
            icon="⚠️",
        )

    if bank_name == "GenericBank":
        st.warning(
            "Unrecognized bank - using generic parser. Check the docs for supported banks "
            "or share feedback so we can train a template.",
            icon="⚠️",
        )

    metadata = _extract_metadata(statement, bank_name)
    return ProcessedFile(pipeline.transform(statement), metadata)


def create_df(processed_files: list[ProcessedFile]) -> pd.DataFrame:
    dataframes = []
    for file in processed_files:
        df = pd.DataFrame(file)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["bank"] = file.metadata.bank_name
        df["currency"] = file.metadata.currency
        df["account_number"] = file.metadata.account_number
        df["statement_start"] = file.metadata.statement_start
        df["statement_end"] = file.metadata.statement_end

        df["credit"] = df["amount"].apply(lambda x: x if x > 0 else 0)
        df["debit"] = df["amount"].apply(lambda x: abs(x) if x < 0 else 0)
        if "polarity" in df.columns:
            normalized = df["polarity"].fillna("").astype(str).str.lower()
            df["is_credit"] = normalized.str.contains("credit")
            df["is_debit"] = normalized.str.contains("debit")
        else:
            df["is_credit"] = df["amount"] > 0
            df["is_debit"] = df["amount"] < 0

        dataframes.append(df)

    concat_df = pd.concat(dataframes)
    st.session_state["df"] = concat_df
    return concat_df


def show_df(df: pd.DataFrame) -> None:
    desired_order = [
        "date",
        "description",
        "amount",
        "debit",
        "credit",
        "bank",
        "account_number",
        "currency",
    ]
    columns_to_use = [col for col in desired_order if col in df.columns]
    df = df[columns_to_use]
    df.columns = [col.title() for col in df.columns]
    st.dataframe(
        df.style.format(
            {
                "Amount": "{:,.2f}",
                "Debit": "{:,.2f}",
                "Credit": "{:,.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    total_balance = df["Amount"].sum()
    st.write(f"Total Balance: ${total_balance:,.2f}")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv,
        mime="text/csv",
    )
