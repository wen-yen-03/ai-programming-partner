"""Shared configuration loading (.env file, environment variables, the
Anthropic client).

One implementation used by every module that needs a secret/connection
string from .env or an Anthropic client (reviewer.py and answering.py both
need ANTHROPIC_API_KEY; vector_store.py needs DATABASE_URL) -- avoids each
module re-implementing its own copy.
"""
from __future__ import annotations

import os

import anthropic

from partner import PROJECT_ROOT


def load_dotenv_if_present() -> None:
    """Load KEY=VALUE lines from the project's .env file into os.environ.

    Minimal stdlib-only loader -- no new dependency for this. Never
    overwrites an already-set environment variable, so a real env var
    always wins over the file. Does nothing if no .env file exists.
    """
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def build_client() -> anthropic.Anthropic:
    """Construct an Anthropic client from ANTHROPIC_API_KEY.

    Loads .env first (if present), then fails fast with a clear error
    rather than a deep SDK stack trace if the key is still missing. Never
    logs the key itself.
    """
    load_dotenv_if_present()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")
    return anthropic.Anthropic()
