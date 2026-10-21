"""Phase 2 — Personal Finance Assistant.

This is where NOOA gets powerful. Phase 1 answered "what category is this one
line?". Phase 2 answers open-ended questions about a WHOLE dataset of
transactions, like:

    "How much did I spend on dining last month?"
    "What were my three biggest expenses, and are any tax-deductible?"
    "Compare my transport spending to my entertainment spending."

New ideas on top of Phase 1
---------------------------
  * PRIVATE STATE: the transaction list lives in `self._transactions`. The LLM
    never sees the raw table — it only sees the *capabilities* we expose.

  * DETERMINISTIC TOOLS: plain Python methods (`list_transactions`,
    `total_by_category`, ...). Because they are normal methods with type hints +
    docstrings, NOOA makes them callable by the model automatically. No separate
    tool-schema JSON to maintain.

  * CodeActStrategy: the `answer()` generation method runs in CodeAct mode, so
    the model ANSWERS BY WRITING PYTHON that calls the tools above and computes
    the result — instead of trying to do arithmetic in its head.

Run it (from the repo root, after setting up .env):

    uv run python -m finance_agent.phase2_assistant
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.strategies import CodeActStrategy

from finance_agent.llm import llm
from finance_agent.phase1_categorizer import Category

@dataclass
class Transaction:
    """One categorized transaction. (In a real app, Phase 1 would produce these.)"""

    date: str  # ISO date, e.g. "2026-09-14"
    merchant: str
    amount: float  # positive = money spent; income is handled via category
    category: Category
    tax_deductible: bool = False

# A tiny sample ledger so the demo runs with zero setup. Swap in your own data,
# or generate it by running each line through the Phase 1 categorizer.
SAMPLE_LEDGER: list[Transaction] = [
    Transaction("2026-09-02", "Costco", 86.10, "groceries"),
    Transaction("2026-09-05", "Uber", 42.00, "transport", tax_deductible=True),
    Transaction("2026-09-08", "Nobu", 230.00, "dining", tax_deductible=True),
    Transaction("2026-09-11", "Spotify", 11.99, "entertainment"),
    Transaction("2026-09-14", "Whole Foods", 54.30, "groceries"),
    Transaction("2026-09-18", "Delta Airlines", 410.00, "travel", tax_deductible=True),
    Transaction("2026-09-21", "AMC Theatres", 32.50, "entertainment"),
    Transaction("2026-09-24", "Shell", 61.20, "transport"),
    Transaction("2026-09-28", "Chipotle", 18.75, "dining"),
]

class FinanceAssistant(Agent, llm=llm):
    """You are a personal finance assistant.

    Answer the user's questions about their spending using ONLY the helper
    methods available to you. Compute figures by calling those methods and doing
    the arithmetic in code — never guess totals. Report money with a leading '$'
    and two decimals, and be concise.
    """

    def __init__(self, transactions: list[Transaction] | None = None) -> None:
        super().__init__()
        # Private application state. The model reaches it only through the
        # typed tool methods below — not by reading this attribute directly.
        self._transactions: list[Transaction] = transactions or list(SAMPLE_LEDGER)

    # ---- Deterministic tools (plain Python the LLM may call) ---------------

    def list_transactions(self) -> list[dict]:
        """Return every transaction as a list of dicts (date, merchant, amount, category)."""
        return [t.__dict__ for t in self._transactions]

    def total_spent(self) -> float:
        """Return the grand total spent across all non-income transactions."""
        return round(sum(t.amount for t in self._transactions if t.category != "income"), 2)

    def total_by_category(self, category: str) -> float:
        """Return the total amount spent in a single category (e.g. 'dining')."""
        return round(sum(t.amount for t in self._transactions if t.category == category), 2)

    def spending_breakdown(self) -> dict[str, float]:
        """Return a {category: total} map across all transactions."""
        out: dict[str, float] = {}
        for t in self._transactions:
            out[t.category] = round(out.get(t.category, 0.0) + t.amount, 2)
        return out

    def deductible_total(self) -> float:
        """Return the total amount of transactions flagged tax_deductible."""
        return round(sum(t.amount for t in self._transactions if t.tax_deductible), 2)

    # ---- Generation method (LLM orchestrates the tools via CodeAct) --------

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=8)))
    async def answer(self, question: str) -> str:
        """Answer the user's finance question using the helper methods on `self`."""
        ...

async def main() -> None:
    agent = FinanceAssistant()

    questions = [
        "How much did I spend in total, and in which category did I spend the most?",
        "How much of my spending is tax-deductible?",
        "Compare my dining spending to my entertainment spending.",
    ]

    for q in questions:
        print(f"
Q: {q}")
        reply = await agent.answer(q)
        print(f"A: {reply}")

if __name__ == "__main__":
    asyncio.run(main())
