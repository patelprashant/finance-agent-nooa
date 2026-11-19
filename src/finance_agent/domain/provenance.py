"""Presentation-neutral transaction rows with complete source provenance.

The builder deliberately walks every transaction from every accepted statement.
Import Review and period exclusions change only each row's status and reason;
they never remove a valid source row from the returned collection. This keeps
excluded rows available to the transaction table and local question selectors
while Dashboard calculations consume only rows marked ``INCLUDED``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from finance_agent.domain.types import (
    ClassificationSource,
    EffectiveClassification,
    ImportExclusion,
    ImportSnapshot,
    InclusionStatus,
    StatementId,
    StatementPeriod,
    TransactionId,
    TransactionRowView,
)

_IMPORT_REVIEW_FALLBACK_REASON = "Excluded during Import Review."


def build_transaction_rows(
    snapshot: ImportSnapshot,
    *,
    classifications: Mapping[TransactionId, EffectiveClassification] | None = None,
    selected_period: StatementPeriod | None = None,
) -> tuple[TransactionRowView, ...]:
    """Build stable rows for all valid transactions in an import snapshot.

    ``classifications`` contains any effective classifications already resolved
    by the corrections layer. A missing entry falls back to the imported values
    so this view can also be built before correction processing exists.

    Import Review exclusion takes precedence over period exclusion. A selected
    period affects only transactions retained by Import Review. The output is
    sorted by immutable source coordinates rather than statement or table input
    order, giving presentation adapters and local selectors a stable row order.
    """

    effective_classifications = classifications or {}
    exclusions = _index_exclusions(snapshot.exclusions)
    rows: list[TransactionRowView] = []

    for statement in snapshot.accepted_statements:
        for transaction in statement.transactions:
            status, reason = _inclusion(
                transaction_id=transaction.transaction_id,
                statement_id=statement.statement_id,
                transaction_date=transaction.transaction_date,
                included_transaction_ids=snapshot.included_transaction_ids,
                exclusions=exclusions,
                selected_period=selected_period,
            )
            classification = effective_classifications.get(
                transaction.transaction_id,
                _imported_classification(
                    transaction.imported_category,
                    transaction.imported_subcategory,
                ),
            )
            rows.append(
                TransactionRowView(
                    transaction_id=transaction.transaction_id,
                    transaction_date=transaction.transaction_date,
                    description=transaction.description,
                    amount=transaction.amount,
                    direction=transaction.direction,
                    classification=classification,
                    account_name=statement.account.account_name,
                    provenance=transaction.provenance,
                    inclusion_status=status,
                    exclusion_reason=reason,
                )
            )

    return tuple(sorted(rows, key=_source_order))


def select_transaction_row(
    rows: Sequence[TransactionRowView], transaction_id: TransactionId
) -> TransactionRowView | None:
    """Return a row by stable transaction ID, including an excluded row."""

    return next((row for row in rows if row.transaction_id == transaction_id), None)


def dashboard_transaction_rows(
    rows: Sequence[TransactionRowView],
) -> tuple[TransactionRowView, ...]:
    """Return only rows eligible for Dashboard calculations.

    This selector does not mutate or replace the complete row collection. Local
    selectors should query the collection returned by :func:`build_transaction_rows`.
    """

    return tuple(
        row for row in rows if row.inclusion_status is InclusionStatus.INCLUDED
    )


def _imported_classification(
    category: str, subcategory: str
) -> EffectiveClassification:
    if not category.strip():
        return EffectiveClassification(
            category="Uncategorized",
            subcategory=subcategory,
            source=ClassificationSource.UNCATEGORIZED,
        )
    return EffectiveClassification(
        category=category,
        subcategory=subcategory,
        source=ClassificationSource.IMPORTED,
    )


def _index_exclusions(
    exclusions: Sequence[ImportExclusion],
) -> tuple[
    Mapping[TransactionId, tuple[str, ...]], Mapping[StatementId, tuple[str, ...]]
]:
    by_transaction: dict[TransactionId, list[str]] = {}
    by_statement: dict[StatementId, list[str]] = {}

    for exclusion in exclusions:
        target = (
            by_statement
            if exclusion.transaction_id is None
            else by_transaction
        )
        key = (
            exclusion.statement_id
            if exclusion.transaction_id is None
            else exclusion.transaction_id
        )
        reasons = target.setdefault(key, [])
        if exclusion.reason and exclusion.reason not in reasons:
            reasons.append(exclusion.reason)

    return (
        {key: tuple(reasons) for key, reasons in by_transaction.items()},
        {key: tuple(reasons) for key, reasons in by_statement.items()},
    )


def _inclusion(
    *,
    transaction_id: TransactionId,
    statement_id: StatementId,
    transaction_date,
    included_transaction_ids: frozenset[TransactionId],
    exclusions: tuple[
        Mapping[TransactionId, tuple[str, ...]], Mapping[StatementId, tuple[str, ...]]
    ],
    selected_period: StatementPeriod | None,
) -> tuple[InclusionStatus, str | None]:
    if transaction_id not in included_transaction_ids:
        transaction_exclusions, statement_exclusions = exclusions
        reasons = transaction_exclusions.get(transaction_id, ())
        if not reasons:
            reasons = statement_exclusions.get(statement_id, ())
        return (
            InclusionStatus.EXCLUDED_IMPORT_REVIEW,
            "; ".join(reasons) if reasons else _IMPORT_REVIEW_FALLBACK_REASON,
        )

    if selected_period is not None and not (
        selected_period.start <= transaction_date <= selected_period.end
    ):
        return (
            InclusionStatus.EXCLUDED_PERIOD,
            (
                f"Transaction date {transaction_date.isoformat()} is outside "
                f"the selected Statement Period "
                f"{selected_period.start.isoformat()} through "
                f"{selected_period.end.isoformat()}."
            ),
        )

    return InclusionStatus.INCLUDED, None


def _source_order(row: TransactionRowView) -> tuple[int, str, int, str]:
    provenance = row.provenance
    return (
        provenance.upload_order,
        str(provenance.statement_id),
        provenance.source_row_number,
        str(row.transaction_id),
    )
