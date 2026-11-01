"""Streamlit web UI — load/upload a CSV, see categorized rows, ask questions.

This is the single page that ties the pipeline and the Phase 2 assistant
together for a human:

    load / upload CSV  ->  load_and_categorize  ->  table + Q&A

Streamlit reruns this whole script on every interaction, so anything expensive
(the categorization pipeline, the assistant) must live behind `st.session_state`
and the on-disk cache — otherwise a button click would re-run the LLM.

Run it (from the repo root, after setting up .env):

    uv run streamlit run src/finance_agent/webapp.py

Categorizing blank-category rows and asking questions need an OpenRouter key in
`.env`; viewing already-categorized data does not.
"""

from __future__ import annotations

import asyncio
import csv
import io
import os
import tempfile
from pathlib import Path

import streamlit as st

# Only the lightweight, key-free modules are imported at module load time. The
# pipeline and the assistant pull in the LLM wiring, so they are imported lazily
# inside the handlers below — that way a missing key shows a warning banner
# instead of crashing the page before it can render (R7.5).
from finance_agent.csv_loader import (
    CsvFormatError,
    default_csv_path,
    load_transactions,
)
from finance_agent.models import CSV_COLUMNS, Transaction


def _has_key() -> bool:
    """Return True if an OpenRouter API key is present in the environment."""
    return bool(os.environ.get("OPENROUTER_API_KEY"))


def _run_async(coro):
    """Drive an async coroutine to completion from Streamlit's sync script."""
    return asyncio.run(coro)


def _validate_header(text: str) -> None:
    """Raise `CsvFormatError` if the uploaded CSV is missing a required column."""
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        raise CsvFormatError("file is empty: no header row found")
    missing = [c for c in CSV_COLUMNS if c not in header]
    if missing:
        raise CsvFormatError(
            f"header missing required column(s): {', '.join(missing)}"
        )


def _categorize(transactions: list[Transaction]) -> list[Transaction]:
    """Run the categorization pipeline over already-parsed transactions."""
    from finance_agent.pipeline import categorize_transactions

    return _run_async(categorize_transactions(transactions))


def _load_sample() -> list[Transaction]:
    """Parse and categorize the bundled sample CSV."""
    return _categorize(load_transactions(default_csv_path()))


def _load_upload(data: bytes) -> list[Transaction]:
    """Validate, parse, and categorize an uploaded CSV file's bytes."""
    text = data.decode("utf-8")
    _validate_header(text)
    # `load_transactions` reads from a path, so stage the upload to a temp file.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8", newline=""
    ) as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    try:
        transactions = load_transactions(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return _categorize(transactions)


def _as_rows(transactions: list[Transaction]) -> list[dict]:
    """Shape transactions into display rows (no pandas; a list of dicts)."""
    return [
        {
            "Date": t.date,
            "Description": t.description,
            "Amount": f"{t.amount:.2f}",
            "Type": "Income" if t.is_income else "Spend",
            "Category": t.category,
            "SubCategory": t.sub_category,
            "Deductible": "✓" if t.tax_deductible else "",
        }
        for t in transactions
    ]


def _ask(transactions: list[Transaction], question: str) -> str:
    """Build a FinanceAssistant over the loaded data and answer one question."""
    from finance_agent.phase2_assistant import FinanceAssistant

    agent = FinanceAssistant(transactions)
    return _run_async(agent.answer(question))


def main() -> None:
    st.set_page_config(page_title="Finance Agent", layout="wide")
    st.title("Finance Agent")

    st.session_state.setdefault("transactions", None)
    st.session_state.setdefault("qa_history", [])

    has_key = _has_key()
    if not has_key:
        st.warning(
            "No OPENROUTER_API_KEY found. You can view CSVs whose rows are "
            "already categorized, but filling in blank categories and asking "
            "questions need the key. Add it to your `.env` and restart."
        )

    # ---- Sidebar: load sample / upload ------------------------------------
    with st.sidebar:
        st.header("Data")
        if st.button("Load sample data"):
            try:
                with st.spinner("Loading and categorizing sample data..."):
                    st.session_state.transactions = _load_sample()
                st.session_state.qa_history = []
            except CsvFormatError as exc:
                st.error(f"Could not load the sample CSV: {exc}")
            except Exception as exc:  # pragma: no cover - surfaced to the UI
                st.error(f"Failed to categorize: {exc}")

        uploaded = st.file_uploader("Upload a CSV", type=["csv"])
        if uploaded is not None and st.button("Load uploaded file"):
            try:
                with st.spinner("Loading and categorizing uploaded data..."):
                    st.session_state.transactions = _load_upload(
                        uploaded.getvalue()
                    )
                st.session_state.qa_history = []
            except CsvFormatError as exc:
                st.error(f"Invalid CSV: {exc}")
            except Exception as exc:  # pragma: no cover - surfaced to the UI
                st.error(f"Failed to categorize: {exc}")

    # ---- Main area: table + Q&A -------------------------------------------
    transactions = st.session_state.transactions
    if not transactions:
        st.info("Load the sample data or upload a CSV from the sidebar to begin.")
        return

    st.subheader("Transactions")
    st.dataframe(_as_rows(transactions), use_container_width=True)

    st.subheader("Ask a question")
    with st.form("ask_form"):
        question = st.text_input("Ask a question about your spending")
        submitted = st.form_submit_button("Submit")

    if submitted and question:
        if not has_key:
            st.error("Asking questions needs an OPENROUTER_API_KEY in your `.env`.")
        else:
            try:
                with st.spinner("Thinking..."):
                    answer = _ask(transactions, question)
                st.session_state.qa_history.insert(0, (question, answer))
            except Exception as exc:  # pragma: no cover - surfaced to the UI
                st.error(f"Could not answer: {exc}")

    for q, a in st.session_state.qa_history:
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
        st.divider()


if __name__ == "__main__":
    main()
