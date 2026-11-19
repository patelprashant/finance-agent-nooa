"""Focused coverage for provenance-preserving transaction row views.

Validates Requirements 1.3, 1.4, 1.5, 2.24, and 3.10.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from finance_agent.domain import (
    AccountId,
    AccountMetadata,
    ClassificationSource,
    EffectiveClassification,
    ImportExclusion,
    ImportSnapshot,
    InclusionStatus,
    ReviewIssueId,
    SourceProvenance,
    StatementId,
    StatementPeriod,
    TransactionDirection,
    TransactionId,
    UploadedStatement,
    build_transaction_rows,
    dashboard_transaction_rows,
    select_transaction_row,
)
from finance_agent.domain.types import ImportedTransaction

pytestmark = pytest.mark.deterministic


def _transaction(
    *,
    transaction_id: str,
    statement_id: str,
    statement_name: str,
    upload_order: int,
    source_row_number: int,
    transaction_date: date,
    category: str = "Groceries",
) -> ImportedTransaction:
    raw = (
        transaction_date.isoformat(),
        f"Description {transaction_id}",
        "10.00",
        "",
        "90.00",
        category,
        "",
    )
    return ImportedTransaction(
        transaction_id=TransactionId(transaction_id),
        transaction_date=transaction_date,
        description=raw[1],
        amount=Decimal("10.00"),
        direction=TransactionDirection.DEBIT,
        balance=Decimal("90.00"),
        imported_category=category,
        imported_subcategory="",
        raw_csv_tuple=raw,
        provenance=SourceProvenance(
            account_id=AccountId(f"account-{statement_id}"),
            statement_id=StatementId(statement_id),
            statement_name=statement_name,
            source_row_number=source_row_number,
            upload_order=upload_order,
        ),
    )


def _statement(
    *,
    statement_id: str,
    statement_name: str,
    account_name: str,
    upload_order: int,
    transactions: tuple[ImportedTransaction, ...],
) -> UploadedStatement:
    return UploadedStatement(
        statement_id=StatementId(statement_id),
        name=statement_name,
        account=AccountMetadata(
            account_id=AccountId(f"account-{statement_id}"),
            account_name=account_name,
            currency="AUD",
        ),
        period=StatementPeriod(date(2026, 9, 1), date(2026, 9, 30)),
        upload_order=upload_order,
        transactions=transactions,
    )


def test_rows_preserve_all_sources_and_stably_label_each_exclusion() -> None:
    period_excluded = _transaction(
        transaction_id="tx-period",
        statement_id="statement-first",
        statement_name="first.csv",
        upload_order=0,
        source_row_number=2,
        transaction_date=date(2026, 8, 31),
    )
    review_excluded = _transaction(
        transaction_id="tx-review",
        statement_id="statement-first",
        statement_name="first.csv",
        upload_order=0,
        source_row_number=3,
        transaction_date=date(2026, 9, 2),
    )
    included = _transaction(
        transaction_id="tx-included",
        statement_id="statement-second",
        statement_name="second.csv",
        upload_order=1,
        source_row_number=2,
        transaction_date=date(2026, 9, 3),
    )
    first = _statement(
        statement_id="statement-first",
        statement_name="first.csv",
        account_name="Savings",
        upload_order=0,
        transactions=(review_excluded, period_excluded),
    )
    second = _statement(
        statement_id="statement-second",
        statement_name="second.csv",
        account_name="Everyday",
        upload_order=1,
        transactions=(included,),
    )
    snapshot = ImportSnapshot(
        accepted_statements=(second, first),
        included_transaction_ids=frozenset(
            {period_excluded.transaction_id, included.transaction_id}
        ),
        exclusions=(
            ImportExclusion(
                transaction_id=review_excluded.transaction_id,
                statement_id=first.statement_id,
                reason="Excluded by duplicate review.",
                issue_id=ReviewIssueId("duplicate-1"),
            ),
        ),
        decisions=(),
    )

    rows = build_transaction_rows(
        snapshot,
        selected_period=StatementPeriod(date(2026, 9, 1), date(2026, 9, 30)),
    )

    assert [row.transaction_id for row in rows] == [
        period_excluded.transaction_id,
        review_excluded.transaction_id,
        included.transaction_id,
    ]
    assert [row.account_name for row in rows] == ["Savings", "Savings", "Everyday"]
    assert [row.provenance.statement_name for row in rows] == [
        "first.csv",
        "first.csv",
        "second.csv",
    ]
    assert [row.provenance.source_row_number for row in rows] == [2, 3, 2]

    assert rows[0].inclusion_status is InclusionStatus.EXCLUDED_PERIOD
    assert "2026-08-31" in (rows[0].exclusion_reason or "")
    assert rows[1].inclusion_status is InclusionStatus.EXCLUDED_IMPORT_REVIEW
    assert rows[1].exclusion_reason == "Excluded by duplicate review."
    assert rows[2].inclusion_status is InclusionStatus.INCLUDED
    assert rows[2].exclusion_reason is None

    assert dashboard_transaction_rows(rows) == (rows[2],)
    assert select_transaction_row(rows, review_excluded.transaction_id) is rows[1]


def test_rows_apply_effective_classification_and_statement_exclusion_reason() -> None:
    transaction = _transaction(
        transaction_id="tx-blank",
        statement_id="statement-1",
        statement_name="statement.csv",
        upload_order=0,
        source_row_number=2,
        transaction_date=date(2026, 9, 2),
        category="   ",
    )
    statement = _statement(
        statement_id="statement-1",
        statement_name="statement.csv",
        account_name="Everyday",
        upload_order=0,
        transactions=(transaction,),
    )
    snapshot = ImportSnapshot(
        accepted_statements=(statement,),
        included_transaction_ids=frozenset(),
        exclusions=(
            ImportExclusion(
                transaction_id=None,
                statement_id=statement.statement_id,
                reason="Statement excluded during overlap review.",
                issue_id=ReviewIssueId("overlap-1"),
            ),
        ),
        decisions=(),
    )
    corrected = EffectiveClassification(
        category="Dining",
        subcategory="Cafe",
        source=ClassificationSource.USER_EDIT,
    )

    corrected_row = build_transaction_rows(
        snapshot,
        classifications={transaction.transaction_id: corrected},
    )[0]
    fallback_row = build_transaction_rows(snapshot)[0]

    assert corrected_row.classification is corrected
    assert corrected_row.exclusion_reason == "Statement excluded during overlap review."
    assert fallback_row.classification == EffectiveClassification(
        category="Uncategorized",
        subcategory="",
        source=ClassificationSource.UNCATEGORIZED,
    )
    assert select_transaction_row((fallback_row,), TransactionId("missing")) is None
