"""Focused tests for task 1.2 immutable typed domain records."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date
from decimal import Decimal

import pytest

from finance_agent.domain import (
    AccountId,
    AccountMetadata,
    AnalysisId,
    AnalysisPhase,
    AnalysisSession,
    ImportedTransaction,
    SourceProvenance,
    StatementId,
    StatementPeriod,
    TransactionCategoryEdit,
    TransactionDirection,
    TransactionId,
    UploadedStatement,
)
from finance_agent.models import Transaction


def _typed_transaction() -> ImportedTransaction:
    provenance = SourceProvenance(
        account_id=AccountId("account-1"),
        statement_id=StatementId("statement-1"),
        statement_name="statement.csv",
        source_row_number=2,
        upload_order=0,
    )
    return ImportedTransaction(
        transaction_id=TransactionId("transaction-1"),
        transaction_date=date(2026, 9, 2),
        description="Corner shop",
        amount=Decimal("12.30"),
        direction=TransactionDirection.DEBIT,
        balance=Decimal("87.70"),
        imported_category="Groceries",
        imported_subcategory="",
        raw_csv_tuple=(
            "2026-09-02",
            "Corner shop",
            "12.30",
            "",
            "87.70",
            "Groceries",
            "",
        ),
        provenance=provenance,
    )


def test_typed_records_keep_decimal_date_and_source_provenance() -> None:
    transaction = _typed_transaction()
    statement = UploadedStatement(
        statement_id=StatementId("statement-1"),
        name="statement.csv",
        account=AccountMetadata(
            account_id=AccountId("account-1"),
            account_name="Everyday",
            currency="AUD",
        ),
        period=StatementPeriod(date(2026, 9, 1), date(2026, 9, 30)),
        upload_order=0,
        transactions=(transaction,),
    )

    assert transaction.transaction_date == date(2026, 9, 2)
    assert transaction.amount == Decimal("12.30")
    assert statement.transactions[0].provenance.statement_name == "statement.csv"
    assert statement.transactions[0].provenance.source_row_number == 2


def test_domain_records_and_session_mappings_are_immutable() -> None:
    transaction = _typed_transaction()
    edit = TransactionCategoryEdit(
        transaction_id=transaction.transaction_id,
        category="Dining",
        subcategory="Cafe",
        revision=1,
    )
    session = AnalysisSession(
        analysis_id=AnalysisId("analysis-1"),
        revision=1,
        phase=AnalysisPhase.READY,
        edits={transaction.transaction_id: edit},
    )

    with pytest.raises(FrozenInstanceError):
        transaction.amount = Decimal("99.00")  # type: ignore[misc]

    with pytest.raises(TypeError):
        session.edits[transaction.transaction_id] = edit  # type: ignore[index, assignment]


def test_legacy_transaction_shape_remains_mutable_and_float_based() -> None:
    transaction = Transaction(
        date="2026-09-02",
        description="Corner shop",
        amount=12.3,
        is_income=False,
        balance=87.7,
    )

    transaction.category = "Groceries"

    assert isinstance(transaction.amount, float)
    assert transaction.date == "2026-09-02"
    assert transaction.category == "Groceries"
