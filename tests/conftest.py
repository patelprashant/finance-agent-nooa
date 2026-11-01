"""Shared pytest fixtures for the finance-agent test suite.

These tests never touch the network or a real LLM. Two things make that safe:

  * The deterministic tool methods and the pipeline logic don't call the model —
    only the fake categorizer below is ever used in place of the real one.
  * Importing `phase2_assistant` resolves the shared `llm` client at import
    time (see `finance_agent.llm`). That resolution doesn't make a network call,
    but it does read the environment. To keep import working even when no real
    `.env`/key is present, we set a dummy `OPENROUTER_API_KEY` *before* any
    finance_agent import, without overwriting a real one if it's already set.

The real `ExpenseCategorizer.categorize()` and `FinanceAssistant.answer()`
generation methods are never called here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Set a placeholder key purely so import-time LLM-client resolution succeeds in
# a clean environment. We do NOT overwrite a real key, and nothing in the tests
# ever makes a network call.
os.environ.setdefault("OPENROUTER_API_KEY", "test-dummy-key")

import pytest

from finance_agent.cache import CategorizationCache
from finance_agent.models import Transaction


@pytest.fixture
def cache_path(tmp_path):
    """A temp path for an on-disk cache file (not yet created)."""
    return tmp_path / "categorization.json"


@pytest.fixture
def cache(cache_path):
    """A fresh `CategorizationCache` backed by a temp file."""
    return CategorizationCache(cache_path)


@pytest.fixture
def sample_transactions() -> list[Transaction]:
    """A small, fully-labelled ledger with hand-computable totals.

    Spend rows (is_income=False):
      dining  12.00  (tax_deductible)
      dining   8.00
      groceries 50.00
      transport 30.00  (tax_deductible)
    Income rows (is_income=True), excluded from spend math:
      income  100.00

    So: total_spent = 100.00; dining total = 20.00;
        breakdown = {dining: 20.00, groceries: 50.00, transport: 30.00};
        deductible_total = 42.00 (12.00 + 30.00).
    """
    return [
        Transaction(
            date="2026-09-01",
            description="Cafe latte",
            amount=12.00,
            is_income=False,
            balance=None,
            category="dining",
            sub_category="coffee",
            tax_deductible=True,
        ),
        Transaction(
            date="2026-09-02",
            description="Bakery",
            amount=8.00,
            is_income=False,
            balance=None,
            category="dining",
            sub_category="pastry",
        ),
        Transaction(
            date="2026-09-03",
            description="Supermarket",
            amount=50.00,
            is_income=False,
            balance=None,
            category="groceries",
        ),
        Transaction(
            date="2026-09-04",
            description="Train ticket",
            amount=30.00,
            is_income=False,
            balance=None,
            category="transport",
            tax_deductible=True,
        ),
        Transaction(
            date="2026-09-05",
            description="Paycheck",
            amount=100.00,
            is_income=True,
            balance=None,
            category="income",
            sub_category="salary",
        ),
    ]


@dataclass
class FakeAnalysis:
    """A plain stand-in for `ExpenseAnalysis`.

    The pipeline only reads these attributes, so we avoid importing the real
    Pydantic model (and its live-LLM wiring) entirely.
    """

    category: str
    sub_category: str
    amount: float
    merchant: str
    is_business: bool
    tax_deductible: bool
    note: str


class FakeCategorizer:
    """A fake categorizer that records how often it was asked to categorize.

    It never touches the network; it just returns a canned `FakeAnalysis` and
    counts calls, so pipeline tests can assert on LLM-call avoidance.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def categorize(self, text: str) -> FakeAnalysis:
        self.calls += 1
        return FakeAnalysis(
            category="shopping",
            sub_category="general",
            amount=0.0,
            merchant="fake-merchant",
            is_business=True,
            tax_deductible=True,
            note="categorized by the fake categorizer",
        )


@pytest.fixture
def fake_categorizer() -> FakeCategorizer:
    """A fresh fake categorizer (call counter starts at zero)."""
    return FakeCategorizer()
