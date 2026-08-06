"""Environment-based configuration (BUILD.md section 1: GROQ_API_KEY,
model tier selection). Loaded once by callers that need a live Groq
client; library code never reads os.environ directly.

Model ids are configurable via GROQ_CHEAP_MODEL / GROQ_STRONG_MODEL so a
model rename or tier swap on Groq's side doesn't require a code change --
the defaults are just the models BUILD.md specifies, current as of this
writing (llama-3.1-8b-instant / llama-3.3-70b-versatile).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

DEFAULT_CHEAP_MODEL = "llama-3.1-8b-instant"
DEFAULT_STRONG_MODEL = "llama-3.3-70b-versatile"


@dataclass(frozen=True)
class GroqSettings:
    """Resolved Groq configuration: credentials plus the two model tiers."""

    api_key: str
    cheap_model: str
    strong_model: str


def load_groq_settings() -> GroqSettings:
    """Load Groq settings from the environment (.env is loaded first if present).

    Raises RuntimeError if GROQ_API_KEY is unset or blank -- fail fast
    rather than construct a client that will 401 on first use.
    """
    load_dotenv()

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Sign up at https://console.groq.com "
            "(no card required) and set it in your environment or a .env file."
        )

    cheap_model = os.environ.get("GROQ_CHEAP_MODEL", "").strip() or DEFAULT_CHEAP_MODEL
    strong_model = os.environ.get("GROQ_STRONG_MODEL", "").strip() or DEFAULT_STRONG_MODEL

    return GroqSettings(api_key=api_key, cheap_model=cheap_model, strong_model=strong_model)
