"""Phase 1 — Smart Expense Categorizer.

The smallest useful NOOA agent. It demonstrates the two core ideas:

  1. A GENERATION METHOD: a method whose body is just `...`. You do NOT write
     the logic — the signature + docstring become the prompt, and the NOOA
     runtime asks the LLM to fulfil the contract.

  2. STRUCTURED OUTPUT: the return type is a Pydantic model. NOOA guarantees you
     get back a VALID, typed object (it auto-retries if the model's first reply
     doesn't fit the schema). No JSON parsing, no `.get("maybe_missing_key")`.

Run it (from the repo root, after setting up .env):

    uv run python -m finance_agent.phase1_categorizer
"""

from __future__ import annotations

import asyncio
from typing import Literal

from nooa import Agent
from pydantic import BaseModel, Field

from finance_agent.llm import llm

# A fixed set of categories keeps results consistent and easy to total up later.
Category = Literal[
    "groceries",
    "dining",
    "transport",
    "shopping",
    "utilities",
    "entertainment",
    "health",
    "travel",
    "income",
    "other",
]

class ExpenseAnalysis(BaseModel):
    """The typed result of categorizing one expense line.

    Every `Field(description=...)` is seen by the model — think of them as
    per-field instructions that steer what goes where.
    """

    category: Category = Field(description="The single best-fit category for this expense.")
    sub_category: str = Field(
        description="A short, specific sub-category within the category "
        "(e.g. 'coffee' under 'dining'); '' if none is obvious."
    )
    amount: float = Field(description="The monetary amount, as a positive number.")
    merchant: str = Field(description="The merchant or payee, if identifiable; else 'unknown'.")
    is_business: bool = Field(description="True if this looks like a business/work expense.")
    tax_deductible: bool = Field(
        description="True if this is plausibly tax-deductible (e.g. business travel, tools)."
    )
    note: str = Field(description="A short one-line explanation of the categorization.")

class ExpenseCategorizer(Agent, llm=llm):
    """You are a meticulous personal-finance assistant that categorizes expenses.

    You are careful with money: you never invent amounts, and when a detail is
    unclear you choose the most conservative reasonable interpretation.
    """

    async def categorize(self, raw_expense: str) -> ExpenseAnalysis:
        """Categorize a single free-text expense line into a structured analysis.

        The input is something a human might jot down, e.g.
        "Uber to airport $42" or "Costco groceries 86.10".
        Extract the amount, infer the merchant and category, and judge whether
        it is a business expense and/or tax-deductible.
        """
        ...  # <- generation method: the LLM implements this from the docstring above

async def main() -> None:
    agent = ExpenseCategorizer()

    examples = [
        "Uber to airport $42 to attend conference",
        "Costco groceries 86.10",
        "Monthly Spotify subscription 11.99",
        "Client dinner at Nobu $230 (took the Acme team out)",
        "Salary deposit 4200",
    ]

    for line in examples:
        result = await agent.categorize(line)
        print(f"\nInput:  {line}")
        print(f"  -> {result.category:<13} ${result.amount:<8} | {result.merchant}")
        print(f"     business={result.is_business}  tax_deductible={result.tax_deductible}")
        print(f"     note: {result.note}")

if __name__ == "__main__":
    asyncio.run(main())
