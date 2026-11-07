"""Exact AUD and statement-date primitives for deterministic analysis.

The typed insights path keeps money as :class:`~decimal.Decimal` and dates as
:class:`~datetime.date`. Parsing deliberately accepts only the narrow source
formats defined by the financial-insights specification.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP

AUD_CENT = Decimal("0.01")

_PLAIN_AUD_PATTERN = re.compile(r"^[0-9]+(?:\.[0-9]{1,2})?$")
_STATEMENT_DATE_PATTERN = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")


class MoneyError(ValueError):
    """Raised when an AUD value violates the typed money contract."""


class DateValueError(ValueError):
    """Raised when a statement date or period violates the date contract."""


def parse_money(raw: str, *, allow_zero: bool = False) -> Decimal:
    """Parse a non-negative plain-decimal AUD amount.

    Accepted values contain digits and, optionally, one or two fractional
    digits. Signs, exponent notation, separators, surrounding whitespace,
    non-finite values, and fractional precision beyond cents are rejected.
    Transaction magnitudes must be greater than zero unless ``allow_zero`` is
    explicitly enabled (for example, when parsing a savings goal).
    """
    if not isinstance(raw, str):
        raise MoneyError("AUD amount must be provided as a plain decimal string")
    if _PLAIN_AUD_PATTERN.fullmatch(raw) is None:
        raise MoneyError(
            "AUD amount must be a non-negative plain decimal with at most two decimal places"
        )

    value = Decimal(raw)
    if not value.is_finite():  # Defensive: the narrow pattern already excludes these.
        raise MoneyError("AUD amount must be finite")
    if value == 0 and not allow_zero:
        raise MoneyError("AUD amount must be greater than zero")
    return value


def _require_finite_decimal(value: Decimal, *, name: str = "value") -> Decimal:
    """Return ``value`` after enforcing the analytical Decimal boundary."""
    if not isinstance(value, Decimal):
        raise TypeError(f"{name} must be a Decimal")
    if not value.is_finite():
        raise MoneyError(f"{name} must be finite")
    return value


def quantize_aud(value: Decimal) -> Decimal:
    """Round an analytical amount to the nearest AUD cent, ties away from zero."""
    return _require_finite_decimal(value).quantize(AUD_CENT, rounding=ROUND_HALF_UP)


def sum_money(values: Iterable[Decimal]) -> Decimal:
    """Return the exact sum of finite Decimal amounts without cent rounding."""
    total = Decimal("0")
    for index, value in enumerate(values):
        total += _require_finite_decimal(value, name=f"values[{index}]")
    return total


def subtract_money(minuend: Decimal, *subtrahends: Decimal) -> Decimal:
    """Subtract finite Decimal amounts exactly without cent rounding."""
    result = _require_finite_decimal(minuend, name="minuend")
    for index, value in enumerate(subtrahends):
        result -= _require_finite_decimal(value, name=f"subtrahends[{index}]")
    return result


def mean_money(values: Iterable[Decimal]) -> Decimal:
    """Return the exact arithmetic mean of one or more finite Decimal amounts."""
    materialized = tuple(values)
    if not materialized:
        raise MoneyError("cannot calculate a money mean from no values")
    return sum_money(materialized) / Decimal(len(materialized))


def format_aud(value: Decimal) -> str:
    """Format a finite analytical amount as canonical ``AUD 0.00`` text."""
    rounded = quantize_aud(value)
    return f"AUD {rounded:.2f}"


def parse_statement_date(raw: str) -> date:
    """Parse exactly ``YYYY-MM-DD`` without accepting broader ISO date forms."""
    if not isinstance(raw, str) or _STATEMENT_DATE_PATTERN.fullmatch(raw) is None:
        raise DateValueError("statement date must use YYYY-MM-DD")
    try:
        parsed = datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError:
        raise DateValueError(f"invalid statement date: {raw!r}") from None

    # ``strptime`` can be permissive on some platforms; the exact round trip
    # keeps this parser constrained to four-digit year and zero-padded fields.
    if parsed.isoformat() != raw:
        raise DateValueError("statement date must use YYYY-MM-DD")
    return parsed


def validate_statement_period(start: date, end: date) -> tuple[date, date]:
    """Validate and return an inclusive statement-period boundary pair.

    Equal boundaries are valid because statement periods are inclusive. This
    function validates dates only; the immutable ``StatementPeriod`` record is
    defined by the shared domain types task.
    """
    if not isinstance(start, date) or isinstance(start, datetime):
        raise DateValueError("statement period start must be a date")
    if not isinstance(end, date) or isinstance(end, datetime):
        raise DateValueError("statement period end must be a date")
    if start > end:
        raise DateValueError("statement period start must be on or before end")
    return start, end


def parse_statement_period(start_raw: str, end_raw: str) -> tuple[date, date]:
    """Parse and validate an inclusive pair of statement-period dates."""
    return validate_statement_period(
        parse_statement_date(start_raw), parse_statement_date(end_raw)
    )
