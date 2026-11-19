"""Statement-period selection and inclusive transaction filtering.

This module operates on a finalized :class:`ImportSnapshot`. It derives the
periods that remain available after Import Review, resolves the sole period
automatically, and requires an explicit choice when several distinct periods
remain. Filtering never mutates the snapshot, so review-approved source rows
remain available even when their dates place them outside the selected
Dashboard period.
"""

from __future__ import annotations

from dataclasses import dataclass

from finance_agent.domain.types import (
    ImportedTransaction,
    ImportSnapshot,
    StatementPeriod,
    TransactionId,
)


class PeriodSelectionError(ValueError):
    """Base error for unavailable or incomplete statement-period selection."""


class MissingStatementPeriodError(PeriodSelectionError):
    """Raised when no active statement contributes a statement period."""


class PeriodSelectionRequiredError(PeriodSelectionError):
    """Raised when several periods exist but no explicit selection was made."""


class UnavailableStatementPeriodError(PeriodSelectionError):
    """Raised when a requested period is not available in the snapshot."""


class PeriodFilterInvariantError(PeriodSelectionError):
    """Raised when retained transaction IDs do not resolve to source rows."""


@dataclass(frozen=True, slots=True)
class PeriodFilterResult:
    """The resolved period and retained transactions inside its boundaries."""

    selected_period: StatementPeriod
    transactions: tuple[ImportedTransaction, ...]


class PeriodFilter:
    """Resolve an analysis period and filter retained rows inclusively."""

    def available_periods(
        self, snapshot: ImportSnapshot
    ) -> tuple[StatementPeriod, ...]:
        """Return distinct periods from statements retained by Import Review.

        A statement-level exclusion is represented by an ``ImportExclusion``
        whose ``transaction_id`` is ``None``. Such a statement does not make
        its period available for Dashboard selection. Transaction-level
        exclusions do not remove the statement or its period.
        """

        excluded_statement_ids = {
            exclusion.statement_id
            for exclusion in snapshot.exclusions
            if exclusion.transaction_id is None
        }
        periods = {
            statement.period
            for statement in snapshot.accepted_statements
            if statement.statement_id not in excluded_statement_ids
        }
        return tuple(sorted(periods, key=lambda period: (period.start, period.end)))

    def select_period(
        self,
        snapshot: ImportSnapshot,
        selected_period: StatementPeriod | None = None,
    ) -> StatementPeriod:
        """Resolve an available period according to the selection rules."""

        available = self.available_periods(snapshot)
        if not available:
            raise MissingStatementPeriodError(
                "Analysis requires an available Statement Period."
            )

        if selected_period is not None:
            if selected_period not in available:
                raise UnavailableStatementPeriodError(
                    "The selected Statement Period is not available in the current analysis."
                )
            return selected_period

        if len(available) == 1:
            return available[0]

        raise PeriodSelectionRequiredError(
            "Select exactly one Statement Period before analysis."
        )

    def filter_transactions(
        self,
        snapshot: ImportSnapshot,
        selected_period: StatementPeriod,
    ) -> tuple[ImportedTransaction, ...]:
        """Return exactly retained rows inside the selected inclusive period."""

        period = self.select_period(snapshot, selected_period)
        transactions_by_id = self._retained_transactions_by_id(snapshot)
        return tuple(
            transaction
            for transaction in transactions_by_id.values()
            if period.start <= transaction.transaction_date <= period.end
        )

    def apply(
        self,
        snapshot: ImportSnapshot,
        selected_period: StatementPeriod | None = None,
    ) -> PeriodFilterResult:
        """Resolve the period and return its inclusive retained transaction set."""

        period = self.select_period(snapshot, selected_period)
        return PeriodFilterResult(
            selected_period=period,
            transactions=self.filter_transactions(snapshot, period),
        )

    @staticmethod
    def _retained_transactions_by_id(
        snapshot: ImportSnapshot,
    ) -> dict[TransactionId, ImportedTransaction]:
        retained_ids = snapshot.included_transaction_ids
        transactions: dict[TransactionId, ImportedTransaction] = {}

        for statement in sorted(
            snapshot.accepted_statements,
            key=lambda item: (item.upload_order, str(item.statement_id)),
        ):
            for transaction in sorted(
                statement.transactions,
                key=lambda item: (
                    item.provenance.source_row_number,
                    str(item.transaction_id),
                ),
            ):
                if transaction.transaction_id in retained_ids:
                    transactions.setdefault(transaction.transaction_id, transaction)

        unresolved = retained_ids.difference(transactions)
        if unresolved:
            unresolved_labels = ", ".join(sorted(map(str, unresolved)))
            raise PeriodFilterInvariantError(
                "Retained transaction IDs are missing from accepted statements: "
                f"{unresolved_labels}."
            )
        return transactions


def available_periods(snapshot: ImportSnapshot) -> tuple[StatementPeriod, ...]:
    """Convenience wrapper for :meth:`PeriodFilter.available_periods`."""

    return PeriodFilter().available_periods(snapshot)


def select_period(
    snapshot: ImportSnapshot,
    selected_period: StatementPeriod | None = None,
) -> StatementPeriod:
    """Convenience wrapper for :meth:`PeriodFilter.select_period`."""

    return PeriodFilter().select_period(snapshot, selected_period)


def filter_transactions(
    snapshot: ImportSnapshot,
    selected_period: StatementPeriod,
) -> tuple[ImportedTransaction, ...]:
    """Convenience wrapper for :meth:`PeriodFilter.filter_transactions`."""

    return PeriodFilter().filter_transactions(snapshot, selected_period)


__all__ = [
    "MissingStatementPeriodError",
    "PeriodFilter",
    "PeriodFilterInvariantError",
    "PeriodFilterResult",
    "PeriodSelectionError",
    "PeriodSelectionRequiredError",
    "UnavailableStatementPeriodError",
    "available_periods",
    "filter_transactions",
    "select_period",
]
