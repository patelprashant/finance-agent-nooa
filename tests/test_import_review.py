"""Focused tests for Import Review detection, decisions, and finalization.

Validates Requirements 2.17 through 2.24.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from finance_agent.domain.types import (
    AccountId,
    AccountMetadata,
    DuplicateChoice,
    ImportedTransaction,
    ReviewIssueKind,
    SourceProvenance,
    StatementId,
    StatementPeriod,
    TransactionDirection,
    TransactionId,
    UploadedStatement,
)
from finance_agent.import_review import ImportReviewIncomplete, ImportReviewService


def _transaction(
    statement_id: str,
    account_id: str,
    transaction_id: str,
    row_number: int,
    transaction_date: date,
    *,
    raw: tuple[str, str, str, str, str, str, str] | None = None,
    upload_order: int = 0,
) -> ImportedTransaction:
    source_tuple = raw or (
        transaction_date.isoformat(),
        transaction_id,
        "10.00",
        "",
        "90.00",
        "Dining",
        "",
    )
    return ImportedTransaction(
        transaction_id=TransactionId(transaction_id),
        transaction_date=transaction_date,
        description=source_tuple[1],
        amount=Decimal(source_tuple[2] or source_tuple[3]),
        direction=(
            TransactionDirection.DEBIT
            if source_tuple[2]
            else TransactionDirection.CREDIT
        ),
        balance=Decimal(source_tuple[4]) if source_tuple[4] else None,
        imported_category=source_tuple[5],
        imported_subcategory=source_tuple[6],
        raw_csv_tuple=source_tuple,
        provenance=SourceProvenance(
            account_id=AccountId(account_id),
            statement_id=StatementId(statement_id),
            statement_name=f"{statement_id}.csv",
            source_row_number=row_number,
            upload_order=upload_order,
        ),
    )


def _statement(
    statement_id: str,
    account_id: str,
    period: tuple[date, date],
    transactions: tuple[ImportedTransaction, ...],
    *,
    upload_order: int = 0,
) -> UploadedStatement:
    return UploadedStatement(
        statement_id=StatementId(statement_id),
        name=f"{statement_id}.csv",
        account=AccountMetadata(
            account_id=AccountId(account_id),
            account_name=f"Account {account_id}",
            currency="AUD",
        ),
        period=StatementPeriod(*period),
        upload_order=upload_order,
        transactions=transactions,
    )


def test_build_detects_date_overlap_and_same_account_duplicate_issues() -> None:
    service = ImportReviewService()
    duplicate_raw = (
        "2026-01-10",
        "Same merchant",
        "10.00",
        "",
        "90.00",
        "Dining",
        "",
    )
    first = _statement(
        "s1",
        "owned",
        (date(2026, 1, 1), date(2026, 1, 31)),
        (
            _transaction("s1", "owned", "t1", 2, date(2026, 1, 10), raw=duplicate_raw),
            _transaction("s1", "owned", "t2", 3, date(2026, 2, 1)),
        ),
    )
    second = _statement(
        "s2",
        "owned",
        (date(2026, 1, 31), date(2026, 2, 28)),
        (_transaction("s2", "owned", "t3", 2, date(2026, 1, 10), raw=duplicate_raw, upload_order=1),),
        upload_order=1,
    )
    other_account = _statement(
        "s3",
        "other",
        (date(2026, 1, 1), date(2026, 1, 31)),
        (_transaction("s3", "other", "t4", 2, date(2026, 1, 10), raw=duplicate_raw, upload_order=2),),
        upload_order=2,
    )

    state = service.build((first, second, other_account))

    date_issues = [item for item in state.issues if item.kind is ReviewIssueKind.DATE_EXCEPTION]
    overlap_issues = [item for item in state.issues if item.kind is ReviewIssueKind.OVERLAP]
    duplicate_issues = [item for item in state.issues if item.kind is ReviewIssueKind.EXACT_DUPLICATE]
    assert [item.transaction_ids for item in date_issues] == [(TransactionId("t2"),), (TransactionId("t3"),)]
    assert len(overlap_issues) == 1
    assert overlap_issues[0].statement_ids == (StatementId("s1"), StatementId("s2"))
    assert overlap_issues[0].overlap_period == StatementPeriod(date(2026, 1, 31), date(2026, 1, 31))
    assert len(duplicate_issues) == 1
    assert duplicate_issues[0].transaction_ids == (TransactionId("t1"), TransactionId("t3"))
    assert tuple(item.statement_name for item in duplicate_issues[0].provenance) == ("s1.csv", "s2.csv")
    assert tuple(item.source_row_number for item in duplicate_issues[0].provenance) == (2, 2)
    assert not service.is_ready(state)


def test_finalization_requires_all_active_decisions_and_applies_them_once() -> None:
    service = ImportReviewService()
    duplicate_raw = (
        "2026-01-10",
        "Same merchant",
        "10.00",
        "",
        "90.00",
        "Dining",
        "",
    )
    statement = _statement(
        "s1",
        "owned",
        (date(2026, 1, 1), date(2026, 1, 31)),
        (
            _transaction("s1", "owned", "later", 4, date(2026, 1, 10), raw=duplicate_raw),
            _transaction("s1", "owned", "survivor", 2, date(2026, 1, 10), raw=duplicate_raw),
            _transaction("s1", "owned", "outside", 3, date(2026, 2, 1)),
        ),
    )
    original = service.build((statement,))

    with pytest.raises(ImportReviewIncomplete) as error:
        service.finalize(original)
    assert set(error.value.issue_ids) == {item.issue_id for item in original.issues}

    date_issue = next(item for item in original.issues if item.kind is ReviewIssueKind.DATE_EXCEPTION)
    after_date = service.decide_date_exception(original, date_issue.issue_id, include=False)
    duplicate_issue = next(item for item in after_date.issues if item.kind is ReviewIssueKind.EXACT_DUPLICATE)
    resolved = service.decide_duplicate(after_date, duplicate_issue.issue_id, DuplicateChoice.INCLUDE_ONCE)

    assert all(item.decision is None for item in original.issues)
    assert service.is_ready(resolved)
    snapshot = service.finalize(resolved)
    assert snapshot.included_transaction_ids == frozenset({TransactionId("survivor")})
    assert {item.transaction_id for item in snapshot.exclusions} == {
        TransactionId("later"),
        TransactionId("outside"),
    }
    assert len(snapshot.exclusions) == 2
    assert all(
        transaction.provenance.statement_name == "s1.csv"
        for transaction in snapshot.accepted_statements[0].transactions
    )


def test_overlap_statement_exclusion_rebuilds_active_duplicates() -> None:
    service = ImportReviewService()
    duplicate_raw = (
        "2026-01-15",
        "Same merchant",
        "10.00",
        "",
        "90.00",
        "Dining",
        "",
    )
    first = _statement(
        "s1",
        "owned",
        (date(2026, 1, 1), date(2026, 1, 31)),
        (_transaction("s1", "owned", "t1", 2, date(2026, 1, 15), raw=duplicate_raw),),
    )
    second = _statement(
        "s2",
        "owned",
        (date(2026, 1, 15), date(2026, 2, 15)),
        (_transaction("s2", "owned", "t2", 2, date(2026, 1, 15), raw=duplicate_raw, upload_order=1),),
        upload_order=1,
    )
    state = service.build((first, second))
    overlap = next(item for item in state.issues if item.kind is ReviewIssueKind.OVERLAP)
    assert any(item.kind is ReviewIssueKind.EXACT_DUPLICATE for item in state.issues)

    resolved = service.decide_overlap(
        state,
        overlap.issue_id,
        {StatementId("s1"): True, StatementId("s2"): False},
    )

    assert not any(item.kind is ReviewIssueKind.EXACT_DUPLICATE for item in resolved.issues)
    assert service.is_ready(resolved)
    snapshot = service.finalize(resolved)
    assert snapshot.included_transaction_ids == frozenset({TransactionId("t1")})
    assert tuple(item.transaction_id for item in snapshot.exclusions) == (TransactionId("t2"),)
    assert snapshot.accepted_statements == (first, second)
