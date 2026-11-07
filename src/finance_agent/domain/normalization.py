"""Local, versioned normalization and deterministic classification helpers.

The functions in this module deliberately use small allowlists and exact token
matching. They do not use an external model or fuzzy matching, so every derived
classification can be reproduced and explained.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

NORMALIZER_VERSION = "merchant-v1"
TRANSFER_EVIDENCE_VERSION = "transfer-evidence-v1"
UNKNOWN_MERCHANT = "unknown"

FEE_CATEGORY_ALIASES = frozenset({"fee", "fees", "bank fee", "bank fees"})
INTEREST_CATEGORY_ALIASES = frozenset({"interest"})
DISCRETIONARY_CATEGORY_ALIASES = frozenset({"dining", "entertainment"})
CASH_WITHDRAWAL_CATEGORY_ALIASES = frozenset(
    {"cash", "cash withdrawal", "cash withdrawals", "atm", "atm withdrawal"}
)

# These are intentionally explicit. A phrase outside this list is not enough
# to raise deterministic transfer evidence.
TRANSFER_TERMS = (
    "account transfer",
    "bank transfer",
    "internal transfer",
    "online transfer",
    "transfer",
    "xfer",
    "tfr",
)

_CASH_WITHDRAWAL_TERMS = ("atm cash withdrawal", "atm withdrawal", "cash withdrawal")
_WHITESPACE_RE = re.compile(r"\s+")

# Common bank-supplied references are removed only when anchored at the end.
# Each pattern requires a digit, and removal is accepted only if recognizable
# merchant text remains before it.
_DESIGNATED_REFERENCE_SUFFIX_RE = re.compile(
    r"\s+(?:ref(?:erence)?|txn|transaction|terminal|term|card)"
    r"\s*(?:(?:no|number)\s*)?[a-z0-9]*(?:\s+[a-z0-9]+)*\d[a-z0-9]*$"
)
_TRAILING_DIGIT_REFERENCE_RE = re.compile(r"\s+(?:x+\s*)?\d{4,18}$")


class TransferEvidenceKind(str, Enum):
    """The deterministic source of transfer evidence."""

    TERM = "term"
    ACCOUNT_IDENTIFIER = "account_identifier"


@dataclass(frozen=True, slots=True)
class NormalizedMerchantIdentity:
    """A normalized merchant coupled to the algorithm version that made it."""

    value: str
    version: str = NORMALIZER_VERSION

    @property
    def merchant(self) -> str:
        """Return the normalized merchant value."""

        return self.value

    @property
    def normalizer_version(self) -> str:
        """Return the version under the domain-model field name."""

        return self.version


@dataclass(frozen=True, slots=True)
class TransferEvidenceItem:
    """One exact transfer term or registered-account token match."""

    kind: TransferEvidenceKind
    value: str
    matched_text: str
    detector_version: str = TRANSFER_EVIDENCE_VERSION


class MerchantNormalizer:
    """Normalize descriptions without removing their recognizable merchant."""

    version = NORMALIZER_VERSION

    def normalize(self, description: str) -> str:
        """Return a stable merchant value for a raw transaction description.

        Unicode is normalized, text is case-folded, punctuation becomes word
        spacing, and only bounded reference suffixes are removed. Leading text
        is never discarded.
        """

        normalized = _normalize_search_text(description)
        if not _contains_letter(normalized):
            return UNKNOWN_MERCHANT

        merchant = normalized
        while True:
            shortened = _strip_reference_suffix(merchant)
            if shortened == merchant:
                break
            merchant = shortened

        # A suffix must never consume the only recognizable merchant text.
        return merchant if _contains_letter(merchant) else normalized

    def identity(self, description: str) -> NormalizedMerchantIdentity:
        """Return the merchant together with this normalizer's version."""

        return NormalizedMerchantIdentity(self.normalize(description), self.version)


def normalize_category_label(value: str | None) -> str:
    """Normalize a category for exact trim-and-case-insensitive comparison."""

    if value is None:
        return ""
    return _WHITESPACE_RE.sub(" ", unicodedata.normalize("NFKC", value).strip()).casefold()


def matches_category_alias(
    category: str | None,
    subcategory: str | None,
    aliases: Iterable[str],
) -> bool:
    """Return whether either classification field exactly matches an alias."""

    normalized_aliases = {normalize_category_label(alias) for alias in aliases}
    return (
        normalize_category_label(category) in normalized_aliases
        or normalize_category_label(subcategory) in normalized_aliases
    )


def is_bank_fee(category: str | None, subcategory: str | None = None) -> bool:
    """Recognize only the fee aliases specified by the requirements."""

    return matches_category_alias(category, subcategory, FEE_CATEGORY_ALIASES)


def is_interest(category: str | None, subcategory: str | None = None) -> bool:
    """Recognize an exact ``interest`` category or subcategory."""

    return matches_category_alias(category, subcategory, INTEREST_CATEGORY_ALIASES)


def discretionary_category(
    category: str | None,
    subcategory: str | None = None,
) -> str | None:
    """Return the canonical discretionary group, with category precedence."""

    for candidate in (category, subcategory):
        normalized = normalize_category_label(candidate)
        if normalized == "dining":
            return "Dining"
        if normalized == "entertainment":
            return "Entertainment"
    return None


def is_discretionary(category: str | None, subcategory: str | None = None) -> bool:
    """Return whether Dining or Entertainment appears in either field."""

    return discretionary_category(category, subcategory) is not None


def is_cash_withdrawal(
    description: str,
    category: str | None = None,
    subcategory: str | None = None,
) -> bool:
    """Detect a cash withdrawal from exact categories or local text phrases."""

    if matches_category_alias(
        category, subcategory, CASH_WITHDRAWAL_CATEGORY_ALIASES
    ):
        return True

    search_text = _normalize_search_text(description)
    return any(_contains_exact_phrase(search_text, term) for term in _CASH_WITHDRAWAL_TERMS)


def detect_transfer_evidence(
    description: str,
    account_identifiers: Iterable[str] = (),
) -> tuple[TransferEvidenceItem, ...]:
    """Return exact transfer-term and registered-account-token evidence.

    Registered identifiers are matched case-insensitively after Unicode and
    whitespace normalization, but are never fuzzily expanded or inferred.
    Blank identifiers are ignored. Overlapping transfer terms prefer the
    longest explicit phrase, so ``internal transfer`` does not also emit the
    generic ``transfer`` evidence.
    """

    search_text = _normalize_exact_text(description)
    evidence_with_spans: list[tuple[int, int, TransferEvidenceItem]] = []

    occupied_term_spans: list[tuple[int, int]] = []
    for term in sorted(TRANSFER_TERMS, key=lambda value: (-len(value), value)):
        for match in _exact_phrase_pattern(term).finditer(search_text):
            span = match.span()
            if any(_spans_overlap(span, occupied) for occupied in occupied_term_spans):
                continue
            occupied_term_spans.append(span)
            evidence_with_spans.append(
                (
                    span[0],
                    span[1],
                    TransferEvidenceItem(
                        kind=TransferEvidenceKind.TERM,
                        value=term,
                        matched_text=match.group(0),
                    ),
                )
            )

    normalized_identifiers = {
        identifier
        for raw_identifier in account_identifiers
        if (identifier := _normalize_exact_text(raw_identifier))
    }
    occupied_identifier_spans: list[tuple[int, int]] = []
    for identifier in sorted(
        normalized_identifiers, key=lambda value: (-len(value), value)
    ):
        for match in _exact_phrase_pattern(identifier).finditer(search_text):
            span = match.span()
            if any(
                _spans_overlap(span, occupied)
                for occupied in occupied_identifier_spans
            ):
                continue
            occupied_identifier_spans.append(span)
            evidence_with_spans.append(
                (
                    span[0],
                    span[1],
                    TransferEvidenceItem(
                        kind=TransferEvidenceKind.ACCOUNT_IDENTIFIER,
                        value=identifier,
                        matched_text=match.group(0),
                    ),
                )
            )

    evidence_with_spans.sort(key=lambda item: (item[0], item[1], item[2].kind.value, item[2].value))
    return tuple(item for _, _, item in evidence_with_spans)


def has_transfer_evidence(
    description: str,
    account_identifiers: Iterable[str] = (),
) -> bool:
    """Return whether any deterministic transfer evidence is present."""

    return bool(detect_transfer_evidence(description, account_identifiers))


def _normalize_search_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    words = [" " if unicodedata.category(character).startswith("P") else character for character in normalized]
    return _WHITESPACE_RE.sub(" ", "".join(words)).strip()


def _normalize_exact_text(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", unicodedata.normalize("NFKC", value).casefold()).strip()


def _contains_letter(value: str) -> bool:
    return any(character.isalpha() for character in value)


def _strip_reference_suffix(value: str) -> str:
    for pattern in (_DESIGNATED_REFERENCE_SUFFIX_RE, _TRAILING_DIGIT_REFERENCE_RE):
        match = pattern.search(value)
        if match is None:
            continue
        prefix = value[: match.start()].strip()
        if _contains_letter(prefix):
            return prefix
    return value


def _contains_exact_phrase(text: str, phrase: str) -> bool:
    return _exact_phrase_pattern(phrase).search(text) is not None


def _exact_phrase_pattern(phrase: str) -> re.Pattern[str]:
    parts = phrase.split(" ")
    escaped = r"\s+".join(re.escape(part) for part in parts)
    return re.compile(rf"(?<!\w){escaped}(?!\w)")


def _spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


__all__ = [
    "CASH_WITHDRAWAL_CATEGORY_ALIASES",
    "DISCRETIONARY_CATEGORY_ALIASES",
    "FEE_CATEGORY_ALIASES",
    "INTEREST_CATEGORY_ALIASES",
    "MerchantNormalizer",
    "NORMALIZER_VERSION",
    "NormalizedMerchantIdentity",
    "TRANSFER_EVIDENCE_VERSION",
    "TRANSFER_TERMS",
    "TransferEvidenceItem",
    "TransferEvidenceKind",
    "UNKNOWN_MERCHANT",
    "detect_transfer_evidence",
    "discretionary_category",
    "has_transfer_evidence",
    "is_bank_fee",
    "is_cash_withdrawal",
    "is_discretionary",
    "is_interest",
    "matches_category_alias",
    "normalize_category_label",
]
