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

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.strategies import CodeActStrategy

from finance_agent.llm import llm
from finance_agent.models import Transaction
from finance_agent.pipeline import load_and_categorize

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
        self._transactions: list[Transaction] = transactions or []

    # ---- Deterministic tools (plain Python the LLM may call) ---------------

    def list_transactions(self) -> list[dict]:
        """Return every transaction as a list of dicts (date, description, amount, category, sub_category)."""
        return [
            {
                "date": t.date,
                "description": t.description,
                "amount": t.amount,
                "category": t.category,
                "sub_category": t.sub_category,
            }
            for t in self._transactions
        ]

    def total_spent(self) -> float:
        """Return the grand total spent across all non-income transactions."""
        return round(sum(t.amount for t in self._transactions if not t.is_income), 2)

    def total_by_category(self, category: str) -> float:
        """Return the total amount spent in a single category (e.g. 'dining')."""
        return round(
            sum(
                t.amount
                for t in self._transactions
                if not t.is_income and t.category == category
            ),
            2,
        )

    def spending_breakdown(self) -> dict[str, float]:
        """Return a {category: total} map across all spend (non-income) transactions."""
        out: dict[str, float] = {}
        for t in self._transactions:
            if t.is_income:
                continue
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
    from finance_agent.csv_loader import default_csv_path

    transactions = await load_and_categorize(default_csv_path())
    agent = FinanceAssistant(transactions)

    questions = [
        "How much did I spend in total, and in which category did I spend the most?",
        "How much of my spending is tax-deductible?",
        "Compare my dining spending to my entertainment spending.",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        reply = await agent.answer(q)
        print(f"A: {reply}")

if __name__ == "__main__":
    asyncio.run(main())
