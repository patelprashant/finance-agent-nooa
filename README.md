# Finance Agent — learning NOOA by building

Two phases: (1) Smart Expense Categorizer, (2) Personal Finance Assistant.
LLM provider: OpenRouter, model `openai/nvidia/nemotron-3.5-super-vl:free`.

Requires Python >=3.12,<3.14 (pinned to match the `nooa` dependency).

## Setup
    uv sync                  # installs deps, including nooa, python-dotenv, mcp
    cp .env.example .env     # then paste your OpenRouter key

## Run
    uv run python -m finance_agent.phase1_categorizer
    uv run python -m finance_agent.phase2_assistant

## Swap models
Edit the `model_name:` line in llm_config.yaml (keep the leading `openai/`).
The part after `openai/` is the OpenRouter model slug, e.g.
`openai/openai/gpt-4o-mini` or `openai/anthropic/claude-3.5-haiku`.
