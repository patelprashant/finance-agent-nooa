"""Tests for the categorization pipeline, with the LLM categorizer mocked.

A fake categorizer (see `conftest.py`) stands in for the real one and records
how many times it was asked to categorize. These prove:

  (a) pre-categorized rows are left untouched and cause zero categorizer calls;
  (b) blank-category rows get categorized and the fields are applied;
  (c) a second run over the same rows with a warm cache makes zero additional
      categorizer calls.

The pipeline functions are async; we drive them with `asyncio.run(...)` so no
pytest plugin is needed.
"""

from __future__ import annotations

import asyncio

from finance_agent.models import Transaction
from finance_agent.pipeline import categorize_transactions


def _precategorized() -> Transaction:
    return Transaction(
        date="2026-09-01",
        description="Costco",
        amount=86.10,
        is_income=False,
        balance=4113.90,
        category="groceries",
        sub_category="bulk",
    )


def _blank() -> Transaction:
    return Transaction(
        date="2026-09-02",
        description="Whole Foods",
        amount=54.30,
        is_income=False,
        balance=3775.61,
        category="",
        sub_category="",
    )


def test_precategorized_rows_untouched_zero_calls(cache, fake_categorizer):
    tx = _precategorized()

    result = asyncio.run(
        categorize_transactions([tx], categorizer=fake_categorizer, cache=cache)
    )

    assert fake_categorizer.calls == 0
    # Existing labels preserved exactly.
    assert result[0].category == "groceries"
    assert result[0].sub_category == "bulk"


def test_blank_rows_get_categorized_and_fields_applied(cache, fake_categorizer):
    tx = _blank()

    result = asyncio.run(
        categorize_transactions([tx], categorizer=fake_categorizer, cache=cache)
    )

    assert fake_categorizer.calls == 1
    applied = result[0]
    # Fields from the fake analysis are applied onto the transaction.
    assert applied.category == "shopping"
    assert applied.sub_category == "general"
    assert applied.is_business is True
    assert applied.tax_deductible is True


def test_warm_cache_second_run_zero_additional_calls(cache, fake_categorizer):
    # First run: the blank row is a true miss -> one categorizer call, written
    # through to the cache.
    first = [_blank()]
    asyncio.run(
        categorize_transactions(first, categorizer=fake_categorizer, cache=cache)
    )
    assert fake_categorizer.calls == 1

    # Second run over an equivalent blank row: the cache is warm, so no extra
    # categorizer calls happen, but the cached fields are still applied.
    second = [_blank()]
    result = asyncio.run(
        categorize_transactions(second, categorizer=fake_categorizer, cache=cache)
    )

    assert fake_categorizer.calls == 1  # unchanged
    assert result[0].category == "shopping"
    assert result[0].sub_category == "general"
