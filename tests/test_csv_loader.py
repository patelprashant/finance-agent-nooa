"""Tests for the CSV loader.

Covers a valid parse and each malformed-row case from R1.3: a missing header
column, a row with both Debit and Credit, a row with neither, and a non-numeric
amount. Every malformed case must raise `CsvFormatError`, and (except the
header case, which is file-level) the message must name the offending row.
"""

from __future__ import annotations

import pytest

from finance_agent.csv_loader import CsvFormatError, load_transactions

HEADER = "Date,Description,Debit,Credit,Balance,Category,SubCategory"


def _write_csv(tmp_path, rows: list[str], *, header: str = HEADER):
    path = tmp_path / "transactions.csv"
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def test_valid_file_parses_expected_records(tmp_path):
    path = _write_csv(
        tmp_path,
        [
            "2026-09-01,Salary,,4200.00,4200.00,income,salary",
            "2026-09-02,Costco,86.10,,4113.90,groceries,bulk",
            "2026-09-03,Uber,42.00,,4071.90,,",
        ],
    )

    txns = load_transactions(path)

    assert len(txns) == 3

    salary, costco, uber = txns

    # Credit -> income
    assert salary.is_income is True
    assert salary.amount == 4200.00
    assert salary.category == "income"
    assert salary.sub_category == "salary"
    assert salary.balance == 4200.00

    # Debit -> spend, with category passed through
    assert costco.is_income is False
    assert costco.amount == 86.10
    assert costco.category == "groceries"
    assert costco.sub_category == "bulk"

    # Blank category passed through as empty string
    assert uber.is_income is False
    assert uber.amount == 42.00
    assert uber.category == ""
    assert uber.sub_category == ""


def test_missing_header_column_raises(tmp_path):
    # Drop the SubCategory column from the header entirely.
    bad_header = "Date,Description,Debit,Credit,Balance,Category"
    path = _write_csv(
        tmp_path,
        ["2026-09-02,Costco,86.10,,4113.90,groceries"],
        header=bad_header,
    )

    with pytest.raises(CsvFormatError) as excinfo:
        load_transactions(path)

    assert "SubCategory" in str(excinfo.value)


def test_row_with_both_debit_and_credit_raises_with_row_number(tmp_path):
    path = _write_csv(
        tmp_path,
        [
            "2026-09-01,Salary,,4200.00,4200.00,income,salary",
            "2026-09-02,Weird,10.00,5.00,100.00,,",  # row 3: both present
        ],
    )

    with pytest.raises(CsvFormatError) as excinfo:
        load_transactions(path)

    message = str(excinfo.value)
    assert "row 3" in message
    assert "Debit/Credit" in message


def test_row_with_neither_debit_nor_credit_raises_with_row_number(tmp_path):
    path = _write_csv(
        tmp_path,
        [
            "2026-09-02,Empty,,,100.00,,",  # row 2: neither present
        ],
    )

    with pytest.raises(CsvFormatError) as excinfo:
        load_transactions(path)

    message = str(excinfo.value)
    assert "row 2" in message
    assert "Debit/Credit" in message


def test_non_numeric_amount_raises_with_row_number(tmp_path):
    path = _write_csv(
        tmp_path,
        [
            "2026-09-01,Salary,,4200.00,4200.00,income,salary",
            "2026-09-02,Costco,abc,,4113.90,groceries,bulk",  # row 3: bad Debit
        ],
    )

    with pytest.raises(CsvFormatError) as excinfo:
        load_transactions(path)

    message = str(excinfo.value)
    assert "row 3" in message
    assert "Debit" in message
