# Finance Agent — learning NOOA by building

Two phases: (1) Smart Expense Categorizer, (2) Personal Finance Assistant.
LLM provider: OpenRouter, default model `openai/gpt-4o-mini`.

## Setup
    uv sync
    cp .env.example .env     # then paste your OpenRouter key

## Run
    uv run python -m finance_agent.phase1_categorizer
    uv run python -m finance_agent.phase2_assistant

## Swap models
Edit the `model_name:` line in llm_config.yaml (keep the leading `openai/`).
