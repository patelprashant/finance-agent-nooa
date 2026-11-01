"""Shared data model for the finance pipeline.

Phase 1 and Phase 2 used to be two standalone demos with their own ad-hoc data
shapes. Now that we load from a CSV and pass the same records through the
categorizer and into the assistant, we need ONE transaction type everyone
agrees on. That lives here, together with the CSV column names so the loader and
(later) the upload validator share a single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass

# CSV column names, in the required order. Defining them here means the loader
# and the UI upload validator check against the same constants.
COL_DATE = "Date"
COL_DESCRIPTION = "Description"
COL_DEBIT = "Debit"
COL_CREDIT = "Credit"
COL_BALANCE = "Balance"
COL_CATEGORY = "Category"
COL_SUBCATEGORY = "SubCategory"

#: The expected header, in order.
CSV_COLUMNS = [
    COL_DATE,
    COL_DESCRIPTION,
    COL_DEBIT,
    COL_CREDIT,
    COL_BALANCE,
    COL_CATEGORY,
    COL_SUBCATEGORY,
]


@dataclass
class Transaction:
    """One transaction, shared by the loader, pipeline, and assistant.

    `Debit`/`Credit` from the CSV are collapsed into `amount` plus `is_income`,
    which is clearer than carrying two nullable columns around the app.
    """

    date: str  # ISO date, e.g. "2026-09-14"
    description: str  # raw text from the CSV
    amount: float  # positive magnitude of the movement
    is_income: bool  # True if this row was a Credit, False if a Debit
    balance: float | None  # running balance, informational
    category: str = ""  # top-level category (filled by the pipeline if blank)
    sub_category: str = ""  # finer category (filled by the pipeline if blank)
    is_business: bool = False
    tax_deductible: bool = False
