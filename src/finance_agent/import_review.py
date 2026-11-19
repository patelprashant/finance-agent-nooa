"""Immutable Import Review detection, decisions, and finalization.

The service keeps accepted source statements intact while deriving the active
review issues from them. Statement exclusions are applied before duplicate
issues are rebuilt, so a duplicate in an excluded overlapping statement never
blocks analysis. Every command returns a new ``ImportReviewState``.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace

from finance_agent.domain.types import (
    DateExceptionDecision,
    DuplicateChoice,
    DuplicateDecision,
    ImportDecision,
    ImportDecisionRecord,
    ImportExclusion,
    ImportReviewIssue,
    ImportReviewState,
    ImportSnapshot,
    ImportedTransaction,
    OverlapDecision,
    OverlapStatementDecision,
    ReviewIssueId,
    ReviewIssueKind,
    StatementId,
    StatementPeriod,
    TransactionId,
    UploadedStatement,
)


class ImportReviewError(ValueError):
    """Base class for invalid Import Review operations."""


class UnknownReviewIssue(ImportReviewError):
    """Raised when a command references no active issue of the expected kind."""


class InvalidReviewDecision(ImportReviewError):
    """Raised when a decision is incomplete or invalid for its issue."""


class ImportReviewIncomplete(ImportReviewError):
    """Raised when finalization is attempted with unresolved active issues."""

    def __init__(self, issue_ids: Sequence[ReviewIssueId]) -> None:
        self.issue_ids = tuple(issue_ids)
        joined = ", ".join(str(issue_id) for issue_id in self.issue_ids)
        super().__init__(f"Import Review has unresolved decisions: {joined}.")


class ImportReviewService:
    """Build and resolve Import Review without mutating source records."""

    def build(
        self, accepted: Sequence[UploadedStatement]
    ) -> ImportReviewState:
        """Detect active date, overlap, and exact-duplicate issues."""
        statements = _validated_statements(accepted)
        return _rebuild_state(statements, ())

    def decide_date_exception(
        self,
        state: ImportReviewState,
        issue_id: ReviewIssueId,
        include: bool,
    ) -> ImportReviewState:
        """Record an explicit include-or-exclude decision for one dated row."""
        issue = _require_issue(state, issue_id, ReviewIssueKind.DATE_EXCEPTION)
        if type(include) is not bool:
            raise InvalidReviewDecision("Date-exception include must be boolean.")
        decision = DateExceptionDecision(issue_id=issue.issue_id, include=include)
        return _with_decision(state, decision)

    def decide_overlap(
        self,
        state: ImportReviewState,
        issue_id: ReviewIssueId,
        decisions: Mapping[StatementId, bool],
    ) -> ImportReviewState:
        """Record an explicit include-or-exclude value for both statements."""
        issue = _require_issue(state, issue_id, ReviewIssueKind.OVERLAP)
        expected = set(issue.statement_ids)
        supplied = set(decisions)
        if supplied != expected:
            missing = sorted(str(item) for item in expected - supplied)
            extra = sorted(str(item) for item in supplied - expected)
            details: list[str] = []
            if missing:
                details.append(f"missing {', '.join(missing)}")
            if extra:
                details.append(f"unexpected {', '.join(extra)}")
            raise InvalidReviewDecision(
                "Overlap decision must cover exactly the affected statements"
                + (f" ({'; '.join(details)})" if details else "")
                + "."
            )
        if any(type(value) is not bool for value in decisions.values()):
            raise InvalidReviewDecision("Overlap include values must be boolean.")

        decision = OverlapDecision(
            issue_id=issue.issue_id,
            statements=tuple(
                OverlapStatementDecision(
                    statement_id=statement_id,
                    include=decisions[statement_id],
                )
                for statement_id in issue.statement_ids
            ),
        )
        return _with_decision(state, decision)

    def decide_duplicate(
        self,
        state: ImportReviewState,
        issue_id: ReviewIssueId,
        choice: DuplicateChoice,
    ) -> ImportReviewState:
        """Record include-once or exclude-all for one active duplicate group."""
        issue = _require_issue(state, issue_id, ReviewIssueKind.EXACT_DUPLICATE)
        try:
            normalized_choice = DuplicateChoice(choice)
        except (TypeError, ValueError) as error:
            raise InvalidReviewDecision(
                "Duplicate choice must be include_once or exclude_both."
            ) from error
        decision = DuplicateDecision(
            issue_id=issue.issue_id,
            choice=normalized_choice,
        )
        return _with_decision(state, decision)

    def unresolved_issue_ids(
        self, state: ImportReviewState
    ) -> tuple[ReviewIssueId, ...]:
        """Return active issues lacking a valid decision in display order."""
        return tuple(
            issue.issue_id
            for issue in state.issues
            if not _decision_matches(issue, issue.decision)
        )

    def is_ready(self, state: ImportReviewState) -> bool:
        """Derive readiness solely from active issue decisions."""
        return not self.unresolved_issue_ids(state)

    def finalize(self, state: ImportReviewState) -> ImportSnapshot:
        """Apply resolved exclusions and return immutable analysis input."""
        unresolved = self.unresolved_issue_ids(state)
        if unresolved:
            raise ImportReviewIncomplete(unresolved)

        active_decisions = {
            record.issue_id: record.decision
            for record in state.decisions
            if any(issue.issue_id == record.issue_id for issue in state.issues)
        }
        excluded_statements, statement_causes = _excluded_statement_causes(
            active_decisions.values()
        )
        transaction_by_id = {
            transaction.transaction_id: transaction
            for statement in state.accepted_statements
            for transaction in statement.transactions
        }
        excluded_ids: set[TransactionId] = set()
        exclusions: list[ImportExclusion] = []

        def exclude(
            transaction: ImportedTransaction,
            reason: str,
            issue_id: ReviewIssueId,
        ) -> None:
            if transaction.transaction_id in excluded_ids:
                return
            excluded_ids.add(transaction.transaction_id)
            exclusions.append(
                ImportExclusion(
                    transaction_id=transaction.transaction_id,
                    statement_id=transaction.provenance.statement_id,
                    reason=reason,
                    issue_id=issue_id,
                )
            )

        for statement in state.accepted_statements:
            if statement.statement_id not in excluded_statements:
                continue
            cause = statement_causes[statement.statement_id]
            for transaction in statement.transactions:
                exclude(
                    transaction,
                    "Statement excluded during overlap review.",
                    cause,
                )

        for issue in state.issues:
            decision = active_decisions[issue.issue_id]
            if (
                issue.kind is ReviewIssueKind.DATE_EXCEPTION
                and isinstance(decision, DateExceptionDecision)
                and not decision.include
            ):
                exclude(
                    transaction_by_id[issue.transaction_ids[0]],
                    "Transaction excluded as an out-of-period date exception.",
                    issue.issue_id,
                )

        for issue in state.issues:
            decision = active_decisions[issue.issue_id]
            if issue.kind is not ReviewIssueKind.EXACT_DUPLICATE:
                continue
            assert isinstance(decision, DuplicateDecision)
            duplicate_transactions = tuple(
                transaction_by_id[transaction_id]
                for transaction_id in issue.transaction_ids
            )
            if decision.choice is DuplicateChoice.INCLUDE_ONCE:
                survivor = min(duplicate_transactions, key=_transaction_sort_key)
                to_exclude = tuple(
                    transaction
                    for transaction in duplicate_transactions
                    if transaction.transaction_id != survivor.transaction_id
                )
                reason = (
                    "Exact duplicate excluded; one deterministic source row "
                    "was retained."
                )
            else:
                to_exclude = duplicate_transactions
                reason = "Exact duplicate group excluded during Import Review."
            for transaction in to_exclude:
                exclude(transaction, reason, issue.issue_id)

        all_transaction_ids = frozenset(transaction_by_id)
        included_transaction_ids = frozenset(all_transaction_ids - excluded_ids)
        exclusions.sort(key=lambda item: _exclusion_sort_key(item, transaction_by_id))
        issue_order = {issue.issue_id: position for position, issue in enumerate(state.issues)}
        final_decisions = tuple(
            sorted(
                (
                    ImportDecisionRecord(issue_id=issue_id, decision=decision)
                    for issue_id, decision in active_decisions.items()
                ),
                key=lambda record: issue_order[record.issue_id],
            )
        )
        return ImportSnapshot(
            accepted_statements=state.accepted_statements,
            included_transaction_ids=included_transaction_ids,
            exclusions=tuple(exclusions),
            decisions=final_decisions,
        )


def _validated_statements(
    accepted: Sequence[UploadedStatement],
) -> tuple[UploadedStatement, ...]:
    statements = tuple(sorted(accepted, key=_statement_sort_key))
    statement_ids = [statement.statement_id for statement in statements]
    if len(statement_ids) != len(set(statement_ids)):
        raise ImportReviewError("Accepted statements must have unique statement IDs.")
    transaction_ids = [
        transaction.transaction_id
        for statement in statements
        for transaction in statement.transactions
    ]
    if len(transaction_ids) != len(set(transaction_ids)):
        raise ImportReviewError("Accepted transactions must have unique transaction IDs.")
    return statements


def _rebuild_state(
    statements: tuple[UploadedStatement, ...],
    records: Sequence[ImportDecisionRecord],
) -> ImportReviewState:
    decisions = {record.issue_id: record.decision for record in records}
    overlap_issues = _detect_overlaps(statements)
    excluded_statements, _ = _excluded_statement_causes(decisions.values())
    active_statements = tuple(
        statement
        for statement in statements
        if statement.statement_id not in excluded_statements
    )
    issues = (
        *_detect_date_exceptions(active_statements),
        *overlap_issues,
        *_detect_exact_duplicates(active_statements),
    )
    active_ids = {issue.issue_id for issue in issues}

    # A changed statement-exclusion decision can split or remove a duplicate
    # group. Keep decisions only for duplicate identities that remain active;
    # a changed group receives a new stable ID and therefore needs a decision.
    retained_records = tuple(
        record
        for record in records
        if not str(record.issue_id).startswith("review-duplicate-")
        or record.issue_id in active_ids
    )
    retained_map = {record.issue_id: record.decision for record in retained_records}
    materialized = tuple(
        replace(issue, decision=retained_map.get(issue.issue_id)) for issue in issues
    )
    return ImportReviewState(
        accepted_statements=statements,
        issues=materialized,
        decisions=tuple(sorted(retained_records, key=lambda item: str(item.issue_id))),
    )


def _with_decision(
    state: ImportReviewState,
    decision: ImportDecision,
) -> ImportReviewState:
    records = [
        record for record in state.decisions if record.issue_id != decision.issue_id
    ]
    records.append(ImportDecisionRecord(issue_id=decision.issue_id, decision=decision))
    return _rebuild_state(state.accepted_statements, records)


def _require_issue(
    state: ImportReviewState,
    issue_id: ReviewIssueId,
    expected_kind: ReviewIssueKind,
) -> ImportReviewIssue:
    issue = next((item for item in state.issues if item.issue_id == issue_id), None)
    if issue is None or issue.kind is not expected_kind:
        raise UnknownReviewIssue(
            f"No active {expected_kind.value} issue exists for {issue_id}."
        )
    return issue


def _decision_matches(
    issue: ImportReviewIssue,
    decision: ImportDecision | None,
) -> bool:
    if decision is None or decision.issue_id != issue.issue_id:
        return False
    if issue.kind is ReviewIssueKind.DATE_EXCEPTION:
        return isinstance(decision, DateExceptionDecision)
    if issue.kind is ReviewIssueKind.EXACT_DUPLICATE:
        return isinstance(decision, DuplicateDecision)
    if not isinstance(decision, OverlapDecision):
        return False
    return {item.statement_id for item in decision.statements} == set(
        issue.statement_ids
    ) and all(type(item.include) is bool for item in decision.statements)


def _detect_date_exceptions(
    statements: Sequence[UploadedStatement],
) -> tuple[ImportReviewIssue, ...]:
    issues: list[ImportReviewIssue] = []
    for statement in statements:
        for transaction in sorted(statement.transactions, key=_transaction_sort_key):
            if statement.period.start <= transaction.transaction_date <= statement.period.end:
                continue
            issue_id = ReviewIssueId(
                "review-date-" + _stable_digest(transaction.transaction_id)
            )
            issues.append(
                ImportReviewIssue(
                    issue_id=issue_id,
                    kind=ReviewIssueKind.DATE_EXCEPTION,
                    statement_ids=(statement.statement_id,),
                    transaction_ids=(transaction.transaction_id,),
                    provenance=(transaction.provenance,),
                )
            )
    return tuple(issues)


def _detect_overlaps(
    statements: Sequence[UploadedStatement],
) -> tuple[ImportReviewIssue, ...]:
    by_account: dict[str, list[UploadedStatement]] = defaultdict(list)
    for statement in statements:
        by_account[str(statement.account.account_id)].append(statement)

    issues: list[ImportReviewIssue] = []
    for account_id in sorted(by_account):
        account_statements = sorted(by_account[account_id], key=_statement_sort_key)
        for left_index, left in enumerate(account_statements):
            for right in account_statements[left_index + 1 :]:
                overlap_start = max(left.period.start, right.period.start)
                overlap_end = min(left.period.end, right.period.end)
                if overlap_start > overlap_end:
                    continue
                statement_ids = (left.statement_id, right.statement_id)
                issue_id = ReviewIssueId(
                    "review-overlap-"
                    + _stable_digest(account_id, *sorted(map(str, statement_ids)))
                )
                issues.append(
                    ImportReviewIssue(
                        issue_id=issue_id,
                        kind=ReviewIssueKind.OVERLAP,
                        statement_ids=statement_ids,
                        transaction_ids=(),
                        provenance=(),
                        overlap_period=StatementPeriod(overlap_start, overlap_end),
                    )
                )
    return tuple(issues)


def _detect_exact_duplicates(
    statements: Sequence[UploadedStatement],
) -> tuple[ImportReviewIssue, ...]:
    groups: dict[
        tuple[str, tuple[str, str, str, str, str, str, str]],
        list[ImportedTransaction],
    ] = defaultdict(list)
    for statement in statements:
        account_id = str(statement.account.account_id)
        for transaction in statement.transactions:
            groups[(account_id, transaction.raw_csv_tuple)].append(transaction)

    duplicate_groups = [
        (key, tuple(sorted(transactions, key=_transaction_sort_key)))
        for key, transactions in groups.items()
        if len(transactions) > 1
    ]
    duplicate_groups.sort(
        key=lambda item: tuple(str(tx.transaction_id) for tx in item[1])
    )

    issues: list[ImportReviewIssue] = []
    for (account_id, raw_tuple), transactions in duplicate_groups:
        transaction_ids = tuple(tx.transaction_id for tx in transactions)
        statement_ids = tuple(
            dict.fromkeys(tx.provenance.statement_id for tx in transactions)
        )
        issue_id = ReviewIssueId(
            "review-duplicate-"
            + _stable_digest(
                account_id,
                _stable_digest(*raw_tuple),
                *map(str, transaction_ids),
            )
        )
        issues.append(
            ImportReviewIssue(
                issue_id=issue_id,
                kind=ReviewIssueKind.EXACT_DUPLICATE,
                statement_ids=statement_ids,
                transaction_ids=transaction_ids,
                provenance=tuple(tx.provenance for tx in transactions),
            )
        )
    return tuple(issues)


def _excluded_statement_causes(
    decisions: Sequence[ImportDecision],
) -> tuple[set[StatementId], dict[StatementId, ReviewIssueId]]:
    causes: dict[StatementId, ReviewIssueId] = {}
    overlap_decisions = sorted(
        (item for item in decisions if isinstance(item, OverlapDecision)),
        key=lambda item: str(item.issue_id),
    )
    for decision in overlap_decisions:
        for statement in decision.statements:
            if not statement.include:
                causes.setdefault(statement.statement_id, decision.issue_id)
    return set(causes), causes


def _statement_sort_key(statement: UploadedStatement) -> tuple[int, str]:
    return statement.upload_order, str(statement.statement_id)


def _transaction_sort_key(
    transaction: ImportedTransaction,
) -> tuple[int, int, str]:
    return (
        transaction.provenance.upload_order,
        transaction.provenance.source_row_number,
        str(transaction.transaction_id),
    )


def _exclusion_sort_key(
    exclusion: ImportExclusion,
    transaction_by_id: Mapping[TransactionId, ImportedTransaction],
) -> tuple[int, int, str]:
    if exclusion.transaction_id is None:
        return (2**31 - 1, 2**31 - 1, str(exclusion.statement_id))
    return _transaction_sort_key(transaction_by_id[exclusion.transaction_id])


def _stable_digest(*parts: object) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        encoded = str(part).encode("utf-8", errors="surrogatepass")
        hasher.update(len(encoded).to_bytes(8, "big"))
        hasher.update(encoded)
    return hasher.hexdigest()[:24]


__all__ = [
    "ImportReviewError",
    "ImportReviewIncomplete",
    "ImportReviewService",
    "InvalidReviewDecision",
    "UnknownReviewIssue",
]
