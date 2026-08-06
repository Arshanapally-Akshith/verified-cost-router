"""Shared pytest setup: load .env once at collection time so tests that
check for GROQ_API_KEY (e.g. the real-API integration test) see a key
set via .env, not just the shell environment.
"""

from dotenv import load_dotenv

load_dotenv()
