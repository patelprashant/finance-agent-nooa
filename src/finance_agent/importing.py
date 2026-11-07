"""In-memory CSV statement validation for the typed insights workflow.

Uploaded content is decoded and parsed directly from bytes. This module never
creates a temporary file and retains raw source tuples only in accepted,
session-scoped domain records so Import Review can detect exact duplicates.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation

from finance_agent.domain.money import (
    DateValueError,
    MoneyError,
    parse_money,
    parse_statement_date,
    validate_statement_period,
)
from finance_agent.domain.types import (
    AccountId,
    AccountMetadata,
    ImportedTransaction,
    SourceProvenance,
    StatementId,
    StatementPeriod,
    StatementValidationResult,
    TransactionDirection,
    TransactionId,
    UploadedStatement,
    ValidationCode,
    ValidationIssue,
)
from finance_agent.models import CSV_COLUMNS

_MIN_STATEMENTS = 1
_MAX_STATEMENTS = 100
_BALANCE_PATTERN = re.compile(r"^-?[0-9]+(?:\.[0-9]{1,2})?$")


class UploadCountError(ValueError):
    """Raised when a batch falls outside the supported statement count."""

    def __init__(self, count: int) -> None:
        self.count = count
        self.minimum = _MIN_STATEMENTS
        self.maximum = _MAX_STATEMENTS
        super().__init__(
            f"Upload must contain between {self.minimum} and {self.maximum} "
            f"CSV statements; received {count}."
        )


@dataclass(frozen=True, slots=True)
class UploadInput:
    """One in-memory statement and its user-supplied account metadata.

    ``account_id`` is optional for compatibility with the initial upload DTO.
    Callers that manage explicit account entries should provide it so multiple
    statements can share an account without relying on its display name. When
    omitted, the validator derives a stable identity from the trimmed account
    name.
    """

    filename: str
    content: bytes
    account_name: str
    currency: str
    period_start: str
    period_end: str
    upload_order: int
    account_id: AccountId | str | None = None


@dataclass(frozen=True, slots=True)
class ImportBatchResult:
    """Statement-level validation results in the submitted batch order."""

    results: tuple[StatementValidationResult, ...]

    @property
    def accepted_statements(self) -> tuple[UploadedStatement, ...]:
        """Return every accepted statement without hiding rejected siblings."""
        return tuple(
            result.accepted for result in self.results if result.accepted is not None
        )

    @property
    def issues(self) -> tuple[ValidationIssue, ...]:
        """Return all diagnostics from all rejected statements."""
        return tuple(issue for result in self.results for issue in result.issues)


@dataclass(frozen=True, slots=True)
class _ParsedRow:
    source_row_number: int
    raw: tuple[str, str, str, str, str, str, str]
    transaction_date: date
    amount: Decimal
    direction: TransactionDirection
    balance: Decimal | None


class ImportValidator:
    """Collect all applicable diagnostics while isolating each statement."""

    def validate_batch(self, uploads: Sequence[UploadInput]) -> ImportBatchResult:
        """Validate 1–100 uploads independently and preserve input ordering."""
        count = len(uploads)
        if not _MIN_STATEMENTS <= count <= _MAX_STATEMENTS:
            raise UploadCountError(count)
        return ImportBatchResult(tuple(self.validate_statement(item) for item in uploads))

    def validate_statement(self, upload: UploadInput) -> StatementValidationResult:
        """Validate one statement completely, accepting it only with no issues."""
        statement_name = _display_name(upload.filename)
        issues: list[ValidationIssue] = []

        account_name = _validate_account_name(upload, statement_name, issues)
        _validate_currency(upload, statement_name, issues)
        period = _validate_period(upload, statement_name, issues)
        decoded = _decode_content(upload, statement_name, issues)

        parsed_rows: list[_ParsedRow] = []
        if decoded is not None:
            parsed_rows = _validate_csv(decoded, statement_name, issues)

        if issues or account_name is None or period is None:
            return StatementValidationResult(accepted=None, issues=tuple(issues))

        statement_id = _statement_id(upload, statement_name, account_name)
        account_id = _account_id(upload, account_name)
        transactions = tuple(
            _to_transaction(row, upload, statement_id, account_id)
            for row in parsed_rows
        )
        accepted = UploadedStatement(
            statement_id=statement_id,
            name=statement_name,
            account=AccountMetadata(
                account_id=account_id,
                account_name=account_name,
                currency="AUD",
            ),
            period=period,
            upload_order=upload.upload_order,
            transactions=transactions,
        )
        return StatementValidationResult(accepted=accepted)


def _display_name(filename: object) -> str:
    """Return inert display metadata without treating a filename as a path."""
    if not isinstance(filename, str) or not filename:
        return "unnamed.csv"
    return filename.replace("\x00", "�")


def _issue(
    code: ValidationCode,
    statement_name: str,
    message: str,
    *,
    field: str | None = None,
    source_row_number: int | None = None,
) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        statement_name=statement_name,
        source_row_number=source_row_number,
        field=field,
        message=message,
    )


def _validate_account_name(
    upload: UploadInput,
    statement_name: str,
    issues: list[ValidationIssue],
) -> str | None:
    if not isinstance(upload.account_name, str) or not upload.account_name.strip():
        issues.append(
            _issue(
                ValidationCode.ACCOUNT_NAME,
                statement_name,
                "Account name must not be empty.",
                field="account_name",
            )
        )
        return None
    return upload.account_name.strip()


def _validate_currency(
    upload: UploadInput,
    statement_name: str,
    issues: list[ValidationIssue],
) -> None:
    if upload.currency != "AUD":
        issues.append(
            _issue(
                ValidationCode.CURRENCY,
                statement_name,
                "Currency must be AUD.",
                field="currency",
            )
        )


def _validate_period(
    upload: UploadInput,
    statement_name: str,
    issues: list[ValidationIssue],
) -> StatementPeriod | None:
    start: date | None = None
    end: date | None = None
    try:
        start = parse_statement_date(upload.period_start)
    except (DateValueError, TypeError):
        issues.append(
            _issue(
                ValidationCode.PERIOD,
                statement_name,
                "Statement period start must be a valid YYYY-MM-DD date.",
                field="period_start",
            )
        )
    try:
        end = parse_statement_date(upload.period_end)
    except (DateValueError, TypeError):
        issues.append(
            _issue(
                ValidationCode.PERIOD,
                statement_name,
                "Statement period end must be a valid YYYY-MM-DD date.",
                field="period_end",
            )
        )

    if start is None or end is None:
        return None
    try:
        validate_statement_period(start, end)
    except DateValueError:
        issues.append(
            _issue(
                ValidationCode.PERIOD,
                statement_name,
                "Statement period start must be on or before end.",
                field="statement_period",
            )
        )
        return None
    return StatementPeriod(start=start, end=end)


def _decode_content(
    upload: UploadInput,
    statement_name: str,
    issues: list[ValidationIssue],
) -> str | None:
    if not isinstance(upload.content, bytes):
        issues.append(
            _issue(
                ValidationCode.UTF8,
                statement_name,
                "Statement content must be UTF-8 encoded bytes.",
                field="content",
            )
        )
        return None
    try:
        return upload.content.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        issues.append(
            _issue(
                ValidationCode.UTF8,
                statement_name,
                f"Statement is not valid UTF-8 at byte {error.start}.",
                field="content",
            )
        )
        return None


def _validate_csv(
    decoded: str,
    statement_name: str,
    issues: list[ValidationIssue],
) -> list[_ParsedRow]:
    reader = csv.reader(io.StringIO(decoded, newline=""), strict=True)
    try:
        header = next(reader, None)
    except csv.Error as error:
        issues.append(
            _issue(
                ValidationCode.ROW_SHAPE,
                statement_name,
                f"CSV header cannot be parsed: {error}.",
                source_row_number=1,
            )
        )
        return []

    _validate_header(header or [], statement_name, issues)
    parsed_rows: list[_ParsedRow] = []
    previous_line_number = reader.line_num

    while True:
        source_row_number = previous_line_number + 1
        try:
            row = next(reader)
        except StopIteration:
            break
        except csv.Error as error:
            issues.append(
                _issue(
                    ValidationCode.ROW_SHAPE,
                    statement_name,
                    f"CSV row cannot be parsed: {error}.",
                    source_row_number=source_row_number,
                )
            )
            break

        previous_line_number = reader.line_num
        parsed = _validate_row(row, statement_name, source_row_number, issues)
        if parsed is not None:
            parsed_rows.append(parsed)

    return parsed_rows


def _validate_header(
    header: list[str],
    statement_name: str,
    issues: list[ValidationIssue],
) -> None:
    expected = tuple(CSV_COLUMNS)

    for column in expected:
        if column not in header:
            issues.append(
                _issue(
                    ValidationCode.HEADER_MISSING,
                    statement_name,
                    f"Missing required column: {column}.",
                    field=column,
                    source_row_number=1,
                )
            )

    seen: set[str] = set()
    for position, column in enumerate(header, start=1):
        if column not in expected or column in seen:
            label = column if column else "<empty>"
            issues.append(
                _issue(
                    ValidationCode.HEADER_EXTRA,
                    statement_name,
                    f"Extra column at position {position}: {label}.",
                    field=column or None,
                    source_row_number=1,
                )
            )
        seen.add(column)

    for expected_position, column in enumerate(expected, start=1):
        if column in header and (
            expected_position > len(header)
            or header[expected_position - 1] != column
        ):
            actual_position = header.index(column) + 1
            issues.append(
                _issue(
                    ValidationCode.HEADER_OUT_OF_ORDER,
                    statement_name,
                    f"Column {column} is at position {actual_position}; "
                    f"expected position {expected_position}.",
                    field=column,
                    source_row_number=1,
                )
            )


def _validate_row(
    row: list[str],
    statement_name: str,
    source_row_number: int,
    issues: list[ValidationIssue],
) -> _ParsedRow | None:
    row_issues_start = len(issues)
    if len(row) != len(CSV_COLUMNS):
        issues.append(
            _issue(
                ValidationCode.ROW_SHAPE,
                statement_name,
                f"Row has {len(row)} columns; expected {len(CSV_COLUMNS)}.",
                source_row_number=source_row_number,
            )
        )

    values = (row + [""] * len(CSV_COLUMNS))[: len(CSV_COLUMNS)]
    raw_date, _description, debit, credit, balance, _category, _subcategory = values

    transaction_date: date | None = None
    try:
        transaction_date = parse_statement_date(raw_date)
    except (DateValueError, TypeError):
        issues.append(
            _issue(
                ValidationCode.DATE,
                statement_name,
                "Date must be a valid YYYY-MM-DD value.",
                field="Date",
                source_row_number=source_row_number,
            )
        )

    debit_present = debit != ""
    credit_present = credit != ""
    if debit_present and credit_present:
        issues.append(
            _issue(
                ValidationCode.DIRECTION,
                statement_name,
                "Exactly one of Debit or Credit must be non-empty.",
                field="Debit/Credit",
                source_row_number=source_row_number,
            )
        )
    elif not debit_present and not credit_present:
        issues.append(
            _issue(
                ValidationCode.MISSING_AMOUNT,
                statement_name,
                "One of Debit or Credit must contain an amount.",
                field="Debit/Credit",
                source_row_number=source_row_number,
            )
        )

    debit_amount = _validate_amount(
        debit, "Debit", debit_present, statement_name, source_row_number, issues
    )
    credit_amount = _validate_amount(
        credit, "Credit", credit_present, statement_name, source_row_number, issues
    )
    parsed_balance = _validate_balance(
        balance, statement_name, source_row_number, issues
    )

    if len(issues) != row_issues_start:
        return None

    assert transaction_date is not None
    if debit_present:
        assert debit_amount is not None
        amount = debit_amount
        direction = TransactionDirection.DEBIT
    else:
        assert credit_amount is not None
        amount = credit_amount
        direction = TransactionDirection.CREDIT

    return _ParsedRow(
        source_row_number=source_row_number,
        raw=tuple(values),  # type: ignore[arg-type]
        transaction_date=transaction_date,
        amount=amount,
        direction=direction,
        balance=parsed_balance,
    )


def _validate_amount(
    raw: str,
    field: str,
    present: bool,
    statement_name: str,
    source_row_number: int,
    issues: list[ValidationIssue],
) -> Decimal | None:
    if not present:
        return None
    try:
        return parse_money(raw)
    except (MoneyError, TypeError):
        issues.append(
            _issue(
                ValidationCode.AMOUNT,
                statement_name,
                f"{field} must be a positive plain AUD amount with at most two decimal places.",
                field=field,
                source_row_number=source_row_number,
            )
        )
        return None


def _validate_balance(
    raw: str,
    statement_name: str,
    source_row_number: int,
    issues: list[ValidationIssue],
) -> Decimal | None:
    if raw == "":
        return None
    if _BALANCE_PATTERN.fullmatch(raw) is None:
        issues.append(
            _issue(
                ValidationCode.BALANCE,
                statement_name,
                "Balance must be a plain numeric AUD value with at most two decimal places.",
                field="Balance",
                source_row_number=source_row_number,
            )
        )
        return None
    try:
        value = Decimal(raw)
    except InvalidOperation:
        value = Decimal("NaN")
    if not value.is_finite():
        issues.append(
            _issue(
                ValidationCode.BALANCE,
                statement_name,
                "Balance must be finite.",
                field="Balance",
                source_row_number=source_row_number,
            )
        )
        return None
    return value


def _stable_digest(*parts: object) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        encoded = str(part).encode("utf-8", errors="surrogatepass")
        hasher.update(len(encoded).to_bytes(8, "big"))
        hasher.update(encoded)
    return hasher.hexdigest()[:24]


def _account_id(upload: UploadInput, account_name: str) -> AccountId:
    if upload.account_id is not None and str(upload.account_id):
        return AccountId(str(upload.account_id))
    return AccountId(f"account-{_stable_digest(account_name)}")


def _statement_id(
    upload: UploadInput, statement_name: str, account_name: str
) -> StatementId:
    return StatementId(
        f"statement-{_stable_digest(upload.upload_order, statement_name, account_name, upload.content)}"
    )


def _to_transaction(
    row: _ParsedRow,
    upload: UploadInput,
    statement_id: StatementId,
    account_id: AccountId,
) -> ImportedTransaction:
    transaction_id = TransactionId(
        f"transaction-{_stable_digest(statement_id, row.source_row_number)}"
    )
    provenance = SourceProvenance(
        account_id=account_id,
        statement_id=statement_id,
        statement_name=_display_name(upload.filename),
        source_row_number=row.source_row_number,
        upload_order=upload.upload_order,
    )
    return ImportedTransaction(
        transaction_id=transaction_id,
        transaction_date=row.transaction_date,
        description=row.raw[1],
        amount=row.amount,
        direction=row.direction,
        balance=row.balance,
        imported_category=row.raw[5],
        imported_subcategory=row.raw[6],
        raw_csv_tuple=row.raw,
        provenance=provenance,
    )


__all__ = [
    "ImportBatchResult",
    "ImportValidator",
    "UploadCountError",
    "UploadInput",
]
