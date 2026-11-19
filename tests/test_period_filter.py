"""Focused unit coverage for statement-period selection and filtering.

Validates Requirements 3.1, 3.2, 3.3, 3.4, and 3.5.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from finance_agent.domain import (
    AccountId,
    AccountMetadata,
    ImportExclusion,
    ImportSnapshot,
    ImportedTransaction,
    ReviewIssueId,
    SourceProvenance,
    StatementId,
    StatementPeriod,
    TransactionDirection,
    TransactionId,
    UploadedStatement,
)
from finance_agent.period_filter import (
    MissingStatementPeriodError,
    PeriodFilter,
    PeriodSelectionRequiredError,
    UnavailableStatementPeriodError,
)


def _transaction(
    identifier: str,
    transaction_date: date,
    statement_id: StatementId,
    row_number: int,
) -> ImportedTransaction:
    return ImportedTransaction(
        transaction_id=TransactionId(identifier),
        transaction_date=transaction_date,
        description=f"Transaction {identifier}",
        amount=Decimal("10.00"),
        direction=TransactionDirection.DEBIT,
        balance=None,
        imported_category="Dining",
        imported_subcategory="",
        raw_csv_tuple=(
            transaction_date.isoformat(),
            f"Transaction {identifier}",
            "10.00",
            "",
            "",
            "Dining",
            "",
        ),
        provenance=SourceProvenance(
            account_id=AccountId("account-1"),
            statement_id=statement_id,
            statement_name=f"{statement_id}.csv",
            source_row_number=row_number,
            upload_order=0,
        ),
    )


def _statement(
    identifier: str,
    period: StatementPeriod,
    transactions: tuple[ImportedTransaction, ...] = (),
    *,
    upload_order: int = 0,
) -> UploadedStatement:
    return UploadedStatement(
        statement_id=StatementId(identifier),
        name=f"{identifier}.csv",
        account=AccountMetadata(
            account_id=AccountId("account-1"),
            account_name="Everyday",
            currency="AUD",
        ),
        period=period,
        upload_order=upload_order,
        transactions=transactions,
    )


def _snapshot(
    statements: tuple[UploadedStatement, ...],
    included_ids: frozenset[TransactionId] = frozenset(),
    exclusions: tuple[ImportExclusion, ...] = (),
) -> ImportSnapshot:
    return ImportSnapshot(
        accepted_statements=statements,
        included_transaction_ids=included_ids,
        exclusions=exclusions,
        decisions=(),
    )


def test_apply_auto_selects_the_sole_distinct_period() -> None:
    period = StatementPeriod(date(2026, 1, 1), date(2026, 1, 31))
    first_id = StatementId("statement-1")
    second_id = StatementId("statement-2")
    first = _transaction("tx-1", date(2026, 1, 5), first_id, 2)
    second = _transaction("tx-2", date(2026, 1, 20), second_id, 2)
    snapshot = _snapshot(
        (
            _statement("statement-1", period, (first,), upload_order=0),
            _statement("statement-2", period, (second,), upload_order=1),
        ),
        frozenset((first.transaction_id, second.transaction_id)),
    )

    result = PeriodFilter().apply(snapshot)

    assert result.selected_period == period
    assert tuple(tx.transaction_id for tx in result.transactions) == (
        first.transaction_id,
        second.transaction_id,
    )


def test_multiple_distinct_periods_require_an_explicit_available_selection() -> None:
    january = StatementPeriod(date(2026, 1, 1), date(2026, 1, 31))
    february = StatementPeriod(date(2026, 2, 1), date(2026, 2, 28))
    snapshot = _snapshot(
        (
            _statement("january", january),
            _statement("february", february, upload_order=1),
        )
    )
    period_filter = PeriodFilter()

    with pytest.raises(PeriodSelectionRequiredError, match="exactly one"):
        period_filter.select_period(snapshot)

    assert period_filter.select_period(snapshot, february) == february

    with pytest.raises(UnavailableStatementPeriodError, match="not available"):
        period_filter.select_period(
            snapshot,
            StatementPeriod(date(2026, 3, 1), date(2026, 3, 31)),
        )


def test_no_active_statement_period_rejects_analysis() -> None:
    period = StatementPeriod(date(2026, 1, 1), date(2026, 1, 31))
    statement = _statement("excluded", period)
    exclusion = ImportExclusion(
        transaction_id=None,
        statement_id=statement.statement_id,
        reason="Statement excluded during overlap review.",
        issue_id=ReviewIssueId("overlap-1"),
    )
    snapshot = _snapshot((statement,), exclusions=(exclusion,))

    with pytest.raises(MissingStatementPeriodError, match="requires"):
        PeriodFilter().select_period(snapshot)


def test_filter_returns_only_retained_transactions_on_inclusive_boundaries() -> None:
    period = StatementPeriod(date(2026, 1, 1), date(2026, 1, 31))
    statement_id = StatementId("statement-1")
    before = _transaction("before", date(2025, 12, 31), statement_id, 2)
    on_start = _transaction("start", period.start, statement_id, 3)
    middle = _transaction("middle", date(2026, 1, 15), statement_id, 4)
    on_end = _transaction("end", period.end, statement_id, 5)
    after = _transaction("after", date(2026, 2, 1), statement_id, 6)
    excluded_inside = _transaction("excluded", date(2026, 1, 10), statement_id, 7)
    statement = _statement(
        "statement-1",
        period,
        (before, on_start, middle, on_end, after, excluded_inside),
    )
    retained = frozenset(
        tx.transaction_id for tx in (before, on_start, middle, on_end, after)
    )
    snapshot = _snapshot((statement,), retained)

    result = PeriodFilter().apply(snapshot)

    assert tuple(tx.transaction_id for tx in result.transactions) == (
        on_start.transaction_id,
        middle.transaction_id,
        on_end.transaction_id,
    )
    assert before.transaction_id in snapshot.included_transaction_ids
    assert after.transaction_id in snapshot.included_transaction_ids
    assert excluded_inside.transaction_id not in snapshot.included_transaction_ids


def test_transaction_level_exclusion_does_not_remove_the_statement_period() -> None:
    period = StatementPeriod(date(2026, 1, 1), date(2026, 1, 31))
    statement_id = StatementId("statement-1")
    transaction = _transaction("excluded", date(2026, 1, 5), statement_id, 2)
    statement = _statement("statement-1", period, (transaction,))
    exclusion = ImportExclusion(
        transaction_id=transaction.transaction_id,
        statement_id=statement.statement_id,
        reason="Transaction excluded during date review.",
        issue_id=ReviewIssueId("date-1"),
    )
    snapshot = _snapshot((statement,), exclusions=(exclusion,))

    result = PeriodFilter().apply(snapshot)

    assert result.selected_period == period
    assert result.transactions == ()
