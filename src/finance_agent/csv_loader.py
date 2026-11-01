"""Load transactions from a CSV file into `Transaction` records.

The whole app reads one CSV shape:

    Date,Description,Debit,Credit,Balance,Category,SubCategory

Each row is either a spend (Debit present) or income (Credit present), never
both and never neither. We validate strictly and, on any problem, raise a
`CsvFormatError` that names the offending row so the error is actionable (and so
the UI can show it cleanly instead of a stack trace).

No network, no API key, stdlib `csv` only.
"""

from __future__ import annotations

import csv
from pathlib import Path

from finance_agent.models import (
    COL_BALANCE,
    COL_CATEGORY,
    COL_CREDIT,
    COL_DATE,
    COL_DEBIT,
    COL_DESCRIPTION,
    COL_SUBCATEGORY,
    CSV_COLUMNS,
    Transaction,
)


class CsvFormatError(Exception):
    """Raised when a CSV file or one of its rows does not match the expected format."""


def default_csv_path() -> Path:
    """Return the path to the bundled sample CSV (`data/sample_transactions.csv`)."""
    return Path(__file__).resolve().parents[2] / "data" / "sample_transactions.csv"


def _parse_float(value: str, *, row_number: int, field: str) -> float:
    """Parse a required numeric field, raising a row-naming error on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        raise CsvFormatError(
            f"row {row_number}: {field} must be numeric, got {value!r}"
        ) from None


def load_transactions(path: str | Path) -> list[Transaction]:
    """Parse a CSV file at `path` into a list of `Transaction` records.

    Rules:
      * The header must contain exactly the expected columns.
      * Each row must have a non-empty `Date`.
      * Exactly one of `Debit`/`Credit` must be present; a `Debit` becomes a
        spend (`is_income=False`), a `Credit` becomes income (`is_income=True`).
      * Amounts and `Balance` must be numeric (`Balance` may be empty).
      * `Category`/`SubCategory` are passed through unchanged (may be blank).

    Any violation raises `CsvFormatError` naming the offending row (1-based,
    counting the header as row 1 so the first data row is row 2).
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        if header is None:
            raise CsvFormatError("file is empty: no header row found")
        missing = [c for c in CSV_COLUMNS if c not in header]
        if missing:
            raise CsvFormatError(
                f"header missing required column(s): {', '.join(missing)}"
            )

        transactions: list[Transaction] = []
        # DictReader's first data row is line 2 of the file (header is line 1).
        for offset, row in enumerate(reader, start=2):
            row_number = offset

            date = (row.get(COL_DATE) or "").strip()
            if not date:
                raise CsvFormatError(f"row {row_number}: Date is required")

            description = (row.get(COL_DESCRIPTION) or "").strip()

            debit = (row.get(COL_DEBIT) or "").strip()
            credit = (row.get(COL_CREDIT) or "").strip()
            if bool(debit) == bool(credit):
                raise CsvFormatError(
                    f"row {row_number}: expected exactly one of Debit/Credit"
                )

            if debit:
                amount = _parse_float(debit, row_number=row_number, field="Debit")
                is_income = False
            else:
                amount = _parse_float(credit, row_number=row_number, field="Credit")
                is_income = True

            balance_raw = (row.get(COL_BALANCE) or "").strip()
            balance = (
                _parse_float(balance_raw, row_number=row_number, field="Balance")
                if balance_raw
                else None
            )

            category = (row.get(COL_CATEGORY) or "").strip()
            sub_category = (row.get(COL_SUBCATEGORY) or "").strip()

            transactions.append(
                Transaction(
                    date=date,
                    description=description,
                    amount=amount,
                    is_income=is_income,
                    balance=balance,
                    category=category,
                    sub_category=sub_category,
                )
            )

    return transactions
