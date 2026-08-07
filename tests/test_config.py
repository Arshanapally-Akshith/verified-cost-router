"""Unit tests for verified_cost_router.config.

`load_dotenv` is monkeypatched to a no-op so these tests are isolated
from whatever is (or isn't) in the real .env file on disk.
"""

from __future__ import annotations

import pytest

from verified_cost_router import config


@pytest.fixture(autouse=True)
def _no_real_dotenv(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "load_dotenv", lambda: None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_CHEAP_MODEL", raising=False)
    monkeypatch.delenv("GROQ_STRONG_MODEL", raising=False)


def test_raises_when_api_key_is_unset(monkeypatch: pytest.MonkeyPatch):
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        config.load_groq_settings()


def test_raises_when_api_key_is_blank(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GROQ_API_KEY", "   ")
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        config.load_groq_settings()


def test_uses_default_models_when_unset(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    settings = config.load_groq_settings()
    assert settings.api_key == "test-key"
    assert settings.cheap_model == config.DEFAULT_CHEAP_MODEL
    assert settings.strong_model == config.DEFAULT_STRONG_MODEL


def test_model_ids_are_overridable_via_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_CHEAP_MODEL", "custom-cheap-model")
    monkeypatch.setenv("GROQ_STRONG_MODEL", "custom-strong-model")

    settings = config.load_groq_settings()

    assert settings.cheap_model == "custom-cheap-model"
    assert settings.strong_model == "custom-strong-model"


def test_blank_model_override_falls_back_to_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_CHEAP_MODEL", "   ")
    settings = config.load_groq_settings()
    assert settings.cheap_model == config.DEFAULT_CHEAP_MODEL


def test_repr_redacts_the_api_key():
    settings = config.GroqSettings(api_key="gsk_super_secret_value_do_not_leak", cheap_model="c", strong_model="s")
    rendered = repr(settings)
    assert "super_secret_value_do_not_leak" not in rendered
    assert "gsk_s" in rendered  # a short prefix is fine for identifying which key
    assert "cheap_model='c'" in rendered
