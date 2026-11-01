"""Tests for the deterministic tool methods on `FinanceAssistant`.

We build the assistant with a fixed ledger and check each tool method against
hand-computed numbers. We never call `answer()` (the generation method), so no
LLM/network is touched. See `conftest.py` for the dummy-key note that keeps the
import-time LLM-client resolution happy in a clean environment.
"""

from __future__ import annotations

from finance_agent.phase2_assistant import FinanceAssistant


def test_total_spent(sample_transactions):
    assistant = FinanceAssistant(sample_transactions)
    # Spend rows only: 12 + 8 + 50 + 30 = 100.00 (income row excluded).
    assert assistant.total_spent() == 100.00


def test_total_by_category(sample_transactions):
    assistant = FinanceAssistant(sample_transactions)
    # dining: 12 + 8 = 20.00
    assert assistant.total_by_category("dining") == 20.00
    assert assistant.total_by_category("groceries") == 50.00
    assert assistant.total_by_category("transport") == 30.00
    # A category with no spend rows totals zero.
    assert assistant.total_by_category("income") == 0.00


def test_spending_breakdown(sample_transactions):
    assistant = FinanceAssistant(sample_transactions)
    assert assistant.spending_breakdown() == {
        "dining": 20.00,
        "groceries": 50.00,
        "transport": 30.00,
    }


def test_deductible_total(sample_transactions):
    assistant = FinanceAssistant(sample_transactions)
    # tax_deductible rows: dining 12.00 + transport 30.00 = 42.00
    assert assistant.deductible_total() == 42.00
