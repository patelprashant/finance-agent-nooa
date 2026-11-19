"""Immutable domain records for the typed financial-insights path.

The legacy :class:`finance_agent.models.Transaction` remains the compatibility
shape used by the existing pipeline.  New analysis code uses the records in
this module so money and dates stay as ``Decimal`` and ``date`` values until a
legacy or presentation boundary explicitly converts them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from types import MappingProxyType
from typing import Literal, Mapping, NewType, TypeVar

AnalysisId = NewType("AnalysisId", str)
AccountId = NewType("AccountId", str)
StatementId = NewType("StatementId", str)
TransactionId = NewType("TransactionId", str)
ReviewIssueId = NewType("ReviewIssueId", str)
FindingId = NewType("FindingId", str)
CitationId = NewType("CitationId", str)

Money = Decimal


class TransactionDirection(str, Enum):
    """Direction of a typed transaction."""

    DEBIT = "debit"
    CREDIT = "credit"


class ValidationCode(str, Enum):
    """Stable codes emitted by statement validation."""

    UTF8 = "utf8"
    ACCOUNT_NAME = "account_name"
    CURRENCY = "currency"
    PERIOD = "period"
    HEADER_MISSING = "header_missing"
    HEADER_EXTRA = "header_extra"
    HEADER_OUT_OF_ORDER = "header_out_of_order"
    ROW_SHAPE = "row_shape"
    DATE = "date"
    DIRECTION = "direction"
    MISSING_AMOUNT = "missing_amount"
    AMOUNT = "amount"
    BALANCE = "balance"


class ReviewIssueKind(str, Enum):
    DATE_EXCEPTION = "date_exception"
    OVERLAP = "overlap"
    EXACT_DUPLICATE = "exact_duplicate"


class DuplicateChoice(str, Enum):
    INCLUDE_ONCE = "include_once"
    EXCLUDE_BOTH = "exclude_both"


class ClassificationSource(str, Enum):
    IMPORTED = "imported"
    USER_EDIT = "user_edit"
    CORRECTION_RULE = "correction_rule"
    UNCATEGORIZED = "uncategorized"


class FindingKind(str, Enum):
    OWNED_TRANSFER = "owned_transfer"
    EXTERNAL_TRANSFER = "external_transfer"
    REFUND = "refund"
    RECURRING = "recurring"
    CLASSIFICATION = "classification"
    SAVINGS_OPPORTUNITY = "savings_opportunity"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FindingStatus(str, Enum):
    CONFIRMED = "confirmed"
    PENDING = "pending"
    REJECTED = "rejected"


class Decision(str, Enum):
    CONFIRM = "confirm"
    REJECT = "reject"


class Cadence(str, Enum):
    WEEKLY = "weekly"
    FORTNIGHTLY = "fortnightly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


class ConsentState(str, Enum):
    NOT_ASKED = "not_asked"
    GRANTED = "granted"
    DECLINED = "declined"


class AnalysisPhase(str, Enum):
    EMPTY = "empty"
    REVIEWING_IMPORT = "reviewing_import"
    READY = "ready"
    REVIEWING_FINDINGS = "reviewing_findings"
    DASHBOARD = "dashboard"
    LEGACY_LOADED = "legacy_loaded"


class InclusionStatus(str, Enum):
    INCLUDED = "included"
    EXCLUDED_IMPORT_REVIEW = "excluded_import_review"
    EXCLUDED_PERIOD = "excluded_period"
    REJECTED_STATEMENT = "rejected_statement"


@dataclass(frozen=True, slots=True)
class StatementPeriod:
    start: date
    end: date


@dataclass(frozen=True, slots=True)
class AccountMetadata:
    account_id: AccountId
    account_name: str
    currency: Literal["AUD"]


@dataclass(frozen=True, slots=True)
class SourceProvenance:
    account_id: AccountId
    statement_id: StatementId
    statement_name: str
    source_row_number: int
    upload_order: int


@dataclass(frozen=True, slots=True)
class ImportedTransaction:
    transaction_id: TransactionId
    transaction_date: date
    description: str
    amount: Decimal
    direction: TransactionDirection
    balance: Decimal | None
    imported_category: str
    imported_subcategory: str
    raw_csv_tuple: tuple[str, str, str, str, str, str, str]
    provenance: SourceProvenance


@dataclass(frozen=True, slots=True)
class UploadedStatement:
    statement_id: StatementId
    name: str
    account: AccountMetadata
    period: StatementPeriod
    upload_order: int
    transactions: tuple[ImportedTransaction, ...]


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: ValidationCode
    statement_name: str
    source_row_number: int | None
    field: str | None
    message: str


@dataclass(frozen=True, slots=True)
class StatementValidationResult:
    accepted: UploadedStatement | None
    issues: tuple[ValidationIssue, ...] = ()


@dataclass(frozen=True, slots=True)
class DateExceptionDecision:
    issue_id: ReviewIssueId
    include: bool


@dataclass(frozen=True, slots=True)
class OverlapStatementDecision:
    statement_id: StatementId
    include: bool


@dataclass(frozen=True, slots=True)
class OverlapDecision:
    issue_id: ReviewIssueId
    statements: tuple[OverlapStatementDecision, ...]


@dataclass(frozen=True, slots=True)
class DuplicateDecision:
    issue_id: ReviewIssueId
    choice: DuplicateChoice


ImportDecision = DateExceptionDecision | OverlapDecision | DuplicateDecision


@dataclass(frozen=True, slots=True)
class ImportReviewIssue:
    issue_id: ReviewIssueId
    kind: ReviewIssueKind
    statement_ids: tuple[StatementId, ...]
    transaction_ids: tuple[TransactionId, ...]
    provenance: tuple[SourceProvenance, ...]
    overlap_period: StatementPeriod | None = None
    decision: ImportDecision | None = None


@dataclass(frozen=True, slots=True)
class ImportDecisionRecord:
    issue_id: ReviewIssueId
    decision: ImportDecision


@dataclass(frozen=True, slots=True)
class ImportExclusion:
    transaction_id: TransactionId | None
    statement_id: StatementId
    reason: str
    issue_id: ReviewIssueId


@dataclass(frozen=True, slots=True)
class ImportReviewState:
    accepted_statements: tuple[UploadedStatement, ...]
    issues: tuple[ImportReviewIssue, ...]
    decisions: tuple[ImportDecisionRecord, ...] = ()


@dataclass(frozen=True, slots=True)
class ImportSnapshot:
    accepted_statements: tuple[UploadedStatement, ...]
    included_transaction_ids: frozenset[TransactionId]
    exclusions: tuple[ImportExclusion, ...]
    decisions: tuple[ImportDecisionRecord, ...]


@dataclass(frozen=True, slots=True)
class EffectiveClassification:
    category: str
    subcategory: str
    source: ClassificationSource


@dataclass(frozen=True, slots=True)
class TransactionCategoryEdit:
    transaction_id: TransactionId
    category: str
    subcategory: str
    revision: int


@dataclass(frozen=True, slots=True)
class CategoryCorrectionRule:
    normalized_merchant: str
    normalizer_version: str
    category: str
    subcategory: str
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class TransferEvidenceItem:
    kind: str
    matched_text: str


@dataclass(frozen=True, slots=True)
class ClassifiedTransaction:
    source: ImportedTransaction
    normalized_merchant: str
    normalizer_version: str
    classification: EffectiveClassification
    transfer_evidence: tuple[TransferEvidenceItem, ...] = ()


@dataclass(frozen=True, slots=True)
class EvidenceInput:
    name: str
    value: str


@dataclass(frozen=True, slots=True)
class RuleEvidence:
    rule_id: str
    rule_version: str
    transaction_ids: tuple[TransactionId, ...]
    inputs: tuple[EvidenceInput, ...]
    calculation: str
    reason: str


@dataclass(frozen=True, slots=True)
class Finding:
    finding_id: FindingId
    kind: FindingKind
    confidence: ConfidenceLevel
    status: FindingStatus
    evidence: RuleEvidence


@dataclass(frozen=True, slots=True)
class TransferFinding(Finding):
    source_transaction_id: TransactionId
    destination_transaction_id: TransactionId | None
    source_account_id: AccountId
    destination: str
    amount: Decimal
    transaction_dates: tuple[date, ...]


@dataclass(frozen=True, slots=True)
class RefundFinding(Finding):
    incoming_transaction_id: TransactionId
    outgoing_transaction_id: TransactionId
    amount: Decimal


@dataclass(frozen=True, slots=True)
class RecurringFinding(Finding):
    normalized_merchant: str
    cadence: Cadence
    occurrence_transaction_ids: tuple[TransactionId, ...]
    occurrence_dates: tuple[date, ...]
    occurrence_amounts: tuple[Decimal, ...]
    annualized_amount: Decimal


@dataclass(frozen=True, slots=True)
class FindingDecision:
    finding_id: FindingId
    decision: Decision
    decision_revision: int


@dataclass(frozen=True, slots=True)
class ReviewQueue:
    pending: tuple[Finding, ...] = ()


@dataclass(frozen=True, slots=True)
class LedgerTerm:
    term_id: str
    transaction_id: TransactionId
    operation: Literal["add", "subtract"]
    amount: Decimal
    reason_code: str
    finding_id: FindingId | None = None


@dataclass(frozen=True, slots=True)
class TotalWithEvidence:
    value: Decimal
    terms: tuple[LedgerTerm, ...]
    expression: str


@dataclass(frozen=True, slots=True)
class GoalState:
    amount: Decimal | None
    met: bool | None
    remaining_gap: Decimal | None


@dataclass(frozen=True, slots=True)
class SavingsOpportunity(Finding):
    category: str
    baseline: TotalWithEvidence
    reduction_percentage: int
    estimated_contribution: Decimal


_MapValue = TypeVar("_MapValue")


def _immutable_mapping(value: Mapping[str, _MapValue]) -> Mapping[str, _MapValue]:
    """Copy a string-keyed mapping so frozen records cannot expose mutable state."""

    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class TotalsSnapshot:
    incoming_funds: TotalWithEvidence
    non_transfer_spending: TotalWithEvidence
    net_cash_flow: TotalWithEvidence
    categories: Mapping[str, TotalWithEvidence]
    discretionary: Mapping[str, TotalWithEvidence]
    fees: TotalWithEvidence
    interest: TotalWithEvidence

    def __post_init__(self) -> None:
        object.__setattr__(self, "categories", _immutable_mapping(self.categories))
        object.__setattr__(self, "discretionary", _immutable_mapping(self.discretionary))


@dataclass(frozen=True, slots=True)
class TransactionRowView:
    transaction_id: TransactionId
    transaction_date: date
    description: str
    amount: Decimal
    direction: TransactionDirection
    classification: EffectiveClassification
    account_name: str
    provenance: SourceProvenance
    inclusion_status: InclusionStatus
    exclusion_reason: str | None = None


@dataclass(frozen=True, slots=True)
class TransferFlowView:
    finding_id: FindingId
    source_account_name: str
    destination: str
    dates: tuple[date, ...]
    amount: Decimal
    confidence: ConfidenceLevel
    evidence: RuleEvidence


@dataclass(frozen=True, slots=True)
class RecurringView:
    finding_id: FindingId
    normalized_merchant: str
    cadence: Cadence
    occurrence_dates: tuple[date, ...]
    occurrence_amounts: tuple[Decimal, ...]
    annualized_amount: Decimal
    confidence: ConfidenceLevel
    evidence: RuleEvidence


@dataclass(frozen=True, slots=True)
class OptionalActionView:
    finding_id: FindingId
    label: str
    estimated_contribution: Decimal
    citation_ids: tuple[CitationId, ...]
    is_optional: Literal[True] = True


@dataclass(frozen=True, slots=True)
class AnswerView:
    text: str
    citation_ids: tuple[CitationId, ...] = ()
    error: str | None = None


@dataclass(frozen=True, slots=True)
class DashboardSnapshot:
    analysis_revision: int
    period: StatementPeriod
    transactions: tuple[TransactionRowView, ...]
    totals: TotalsSnapshot | None
    transfer_flows: tuple[TransferFlowView, ...]
    recurring_commitments: tuple[RecurringView, ...]
    findings: tuple[Finding, ...]
    review_queue: ReviewQueue
    goal: GoalState
    opportunities: tuple[SavingsOpportunity, ...]
    optional_actions: tuple[OptionalActionView, ...]
    empty_explanation: str | None
    safety_notice: str


@dataclass(frozen=True, slots=True)
class AnalysisSession:
    analysis_id: AnalysisId
    revision: int
    phase: AnalysisPhase
    validation_results: tuple[StatementValidationResult, ...] = ()
    import_review: ImportReviewState | None = None
    import_snapshot: ImportSnapshot | None = None
    selected_period: StatementPeriod | None = None
    edits: Mapping[TransactionId, TransactionCategoryEdit] = field(default_factory=dict)
    finding_decisions: Mapping[FindingId, FindingDecision] = field(default_factory=dict)
    goal_amount: Decimal | None = None
    dashboard: DashboardSnapshot | None = None
    consent: ConsentState = ConsentState.NOT_ASKED
    qa_history: tuple[AnswerView, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "edits", MappingProxyType(dict(self.edits)))
        object.__setattr__(
            self, "finding_decisions", MappingProxyType(dict(self.finding_decisions))
        )
