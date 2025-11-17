import hashlib

import pandas as pd
import streamlit as st
from monopoly.pdf import MissingPasswordError, PdfDocument
from streamlit.runtime.uploaded_file_manager import UploadedFile

from webapp.constants import APP_DESCRIPTION
from webapp.helpers import create_df, parse_bank_statement, show_df
from webapp.logo import logo
from webapp.models import ProcessedFile

# number of files that need to be added before progress bar appears
PBAR_MIN_FILES = 4


def app() -> pd.DataFrame:
    st.set_page_config(page_title="Statement Sensei", layout="wide")
    st.image(logo, width=450)
    st.markdown(APP_DESCRIPTION)

    files = get_files()

    df = None
    if "df" in st.session_state:
        df = st.session_state["df"]

    if files:
        processed_files = process_files(files)

        if processed_files:
            df = create_df(processed_files)

    if df is not None:
        show_df(df)
        st.info(
            "Jump into the dashboard for interactive filters and cash-flow charts.",
            icon="📊",
        )
        if st.button("Open dashboard", key="open-dashboard"):
            st.switch_page("pages/1_visualizations.py")

    return df


def process_files(uploaded_files: list[UploadedFile]) -> list[ProcessedFile] | None:
    num_files = len(uploaded_files)
    show_pbar = num_files > PBAR_MIN_FILES

    pbar = st.progress(0, text="Processing PDFs") if show_pbar else None

    processed_files = []
    for i, file in enumerate(uploaded_files):
        if pbar:
            pbar.progress(i / num_files, text=f"Processing {file.name}")

        file_bytes = file.getvalue()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        document = PdfDocument(file_bytes=file_bytes)
        document._name = file.name

        # attempt to use passwords stored in environment to unlock
        # if no passwords in environment, then ask user for password
        password: str | None = None
        if document.is_encrypted:  # pylint: disable=no-member
            try:
                document = document.unlock_document()

            except MissingPasswordError:
                document, password = handle_encrypted_document(document)

        if document:
            processed_file = handle_file(document, file_hash, password=password)
            if processed_file:
                processed_files.append(processed_file)

    if pbar:
        pbar.empty()

    return processed_files


def handle_file(
    document: PdfDocument, cache_key: str | None, password: str | None = None
) -> ProcessedFile | None:
    cache_keys: list[str] = []
    if cache_key:
        cache_keys.append(cache_key)

    document_id = None
    try:
        document_id = document.xref_get_key(-1, "ID")[-1]
    except (AttributeError, TypeError, IndexError):
        document_id = None

    if document_id:
        cache_keys.append(document.name + document_id)

    for key in cache_keys:
        if key in st.session_state:
            return st.session_state[key]

    try:
        file = parse_bank_statement(document, password=password)
    except Exception as exc:  # pylint: disable=broad-except
        st.error(
            f"Failed to parse {document.name}: {exc}",
            icon="❌",
        )
        return None

    for key in cache_keys:
        st.session_state[key] = file
    return file


def handle_encrypted_document(
    document: PdfDocument,
) -> tuple[PdfDocument | None, str | None]:
    passwords: list[str] = st.session_state.setdefault("pdf_passwords", [])

    # Try existing passwords first
    for password in passwords:
        document.authenticate(password)
        if not document.is_encrypted:  # pylint: disable=no-member
            return document, password

    # Prompt user for password if none of the existing passwords work
    password_container = st.empty()
    password = password_container.text_input(
        label="Password",
        type="password",
        placeholder=f"Enter password for {document.name}",
        key=document.name,
    )

    if not password:
        return None, None

    document.authenticate(password)

    if not document.is_encrypted:  # pylint: disable=no-member
        passwords.append(password)
        password_container.empty()
        return document, password

    st.error("Wrong password. Please try again.")
    return None, None


def get_files() -> list[UploadedFile]:
    return st.file_uploader(
        label="Upload a bank statement",
        type="pdf",
        label_visibility="hidden",
        accept_multiple_files=True,
    )


if __name__ == "__main__":
    app()
