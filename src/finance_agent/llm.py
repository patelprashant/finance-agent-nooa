"""Shared LLM wiring for every phase of this project.

Why this file exists
--------------------
The NOOA quickstart examples import `from nooa.util.quickstart import *`, which
guesses a model from whatever provider key happens to be in your environment.
That is handy for the framework's own demos, but for a real project we want to
be explicit: ONE model, chosen on purpose, defined in `llm_config.yaml`.

So here we:
  1. load `.env` (your OPENROUTER_API_KEY + NEMO_OO_LLM_CONFIG),
  2. hand NOOA the path to our registry file, and
  3. resolve the "finance-llm" alias into a client the agents can use.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env BEFORE importing/using NOOA's registry, so OPENROUTER_API_KEY and
# NEMO_OO_LLM_CONFIG are visible when the registry resolves our alias.
load_dotenv(override=True)

# If the user didn't set NEMO_OO_LLM_CONFIG in .env, default it to the
# llm_config.yaml that ships next to this project. This makes the repo work
# out-of-the-box after just setting OPENROUTER_API_KEY.
_default_cfg = Path(__file__).resolve().parents[2] / "llm_config.yaml"
os.environ.setdefault("NEMO_OO_LLM_CONFIG", str(_default_cfg))

from nooa.unifiedllm.registry import get_llm_client  # noqa: E402  (import after env setup)

# The alias name must match the key under `models:` in llm_config.yaml.
MODEL_ALIAS = "finance-llm"

#: A ready-to-use LLM client. Import this into any agent: `llm=llm`.
llm = get_llm_client(MODEL_ALIAS)
