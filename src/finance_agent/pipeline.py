"""The pipeline — connect the loader, the Phase 1 categorizer, and the cache.

This is the "connect phase 1 to phase 2" piece. Given a list of transactions
(typically straight from the CSV loader), it fills in the category and flags for
any row whose `Category` is blank, respecting rows that already carry a label
and skipping the LLM entirely for rows already in the cache.

`categorize_transactions` takes the categorizer and cache as parameters so tests
can inject a fake categorizer (no network) and a temp-file cache.
"""

from __future__ import annotations

from pathlib import Path

from finance_agent.cache import CategorizationCache
from finance_agent.csv_loader import load_transactions
from finance_agent.models import Transaction


async def categorize_transactions(
    transactions: list[Transaction],
    categorizer=None,
    cache: CategorizationCache | None = None,
) -> list[Transaction]:
    """Fill category/sub_category/flags on rows whose `Category` is blank.

    Rows that already have a non-empty `Category` are left untouched (no LLM
    call). Blank-category rows are checked against the cache first; only true
    misses call `categorizer.categorize(...)`, and each fresh result is written
    back to the cache.

    `categorizer` and `cache` are injectable for testing. When omitted, a real
    `ExpenseCategorizer` and the default on-disk cache are created lazily — the
    categorizer is only imported/constructed if at least one row needs it, so
    fully pre-categorized data never touches the LLM wiring.
    """
    if cache is None:
        cache = CategorizationCache()

    for tx in transactions:
        # Respect existing labels.
        if tx.category:
            continue

        key = cache.key(tx.description, tx.amount)
        cached = cache.get(key)
        if cached is not None:
            _apply(tx, cached)
            continue

        # True miss — we need the categorizer. Create a real one lazily if the
        # caller didn't inject one.
        if categorizer is None:
            from finance_agent.phase1_categorizer import ExpenseCategorizer

            categorizer = ExpenseCategorizer()

        analysis = await categorizer.categorize(
            f"{tx.description} {tx.amount:.2f}"
        )
        value = {
            "category": analysis.category,
            "sub_category": analysis.sub_category,
            "is_business": analysis.is_business,
            "tax_deductible": analysis.tax_deductible,
            "note": analysis.note,
        }
        cache.set(key, value)
        _apply(tx, value)

    return transactions


def _apply(tx: Transaction, value: dict) -> None:
    """Apply a categorization result (dict) onto a transaction in place."""
    tx.category = value.get("category", tx.category)
    tx.sub_category = value.get("sub_category", tx.sub_category)
    tx.is_business = value.get("is_business", tx.is_business)
    tx.tax_deductible = value.get("tax_deductible", tx.tax_deductible)


async def load_and_categorize(path: str | Path) -> list[Transaction]:
    """Load a CSV at `path` and return its transactions, fully categorized."""
    transactions = load_transactions(path)
    return await categorize_transactions(transactions)
