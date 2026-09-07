"""
Tests for the optional Gemini-backed conversational layer. These run
WITHOUT a real API key (none is configured in the test environment) to
verify the safe-by-default fallback behavior, plus mocked-client tests
that verify the integration logic itself -- system prompt wiring and
off-topic sentinel handling -- without making a real network call.
"""
import os
from unittest.mock import MagicMock

import gemini_chat
from chatbot import generate_reply


def _clear_gemini_state():
    os.environ.pop("GEMINI_API_KEY", None)
    gemini_chat._client = None


def test_no_api_key_returns_none():
    _clear_gemini_state()
    assert gemini_chat.get_llm_reply("I have a headache") is None


def test_chatbot_falls_back_to_rule_based_without_api_key(client):
    _clear_gemini_state()
    resp = client.post("/chatbot/message", json={"message": "I have a fever and headache"})
    data = resp.get_json()
    assert data["type"] == "matches"


def test_emergency_detection_never_touches_llm(monkeypatch):
    """The hardcoded red-flag check must short-circuit before the LLM is
    ever consulted -- verified by making any LLM call raise, and
    confirming the emergency response still comes back correctly."""
    def _boom(*a, **kw):
        raise AssertionError("LLM should never be called for a red-flag message")
    monkeypatch.setattr(gemini_chat, "get_llm_reply", _boom)

    result = generate_reply("I have chest pain and difficulty breathing")
    assert result["type"] == "urgent"


class _FakeResponse:
    def __init__(self, text):
        self.text = text


def test_mocked_on_topic_reply_passes_through(monkeypatch):
    _clear_gemini_state()
    os.environ["GEMINI_API_KEY"] = "fake-key-for-test"

    def fake_generate_content(model, contents, config):
        assert config.system_instruction == gemini_chat.SYSTEM_PROMPT
        return _FakeResponse("This could be a tension headache; try resting in a dark room.")

    fake_client = MagicMock()
    fake_client.models.generate_content = fake_generate_content
    gemini_chat._client = fake_client

    reply = gemini_chat.get_llm_reply("I have a headache, what could it be?")
    assert reply is not None
    assert "headache" in reply.lower()
    _clear_gemini_state()


def test_thinking_config_matches_model_generation():
    """Regression test: Gemini 2.x and Gemini 3.x use two different,
    MUTUALLY EXCLUSIVE thinking-control parameters -- sending the wrong
    one is a hard 400 error, confirmed against the real API while
    building this (switching the default model from 2.5 to 3.6 broke
    the chatbot until this generation-detection was added). Every
    supported generation must map to its own correct parameter."""
    from google.genai import types

    tc_25 = gemini_chat._build_thinking_config("gemini-2.5-flash", types)
    assert tc_25.thinking_budget == 0
    assert tc_25.thinking_level is None

    tc_15 = gemini_chat._build_thinking_config("gemini-1.5-flash", types)
    assert tc_15.thinking_budget == 0

    tc_36 = gemini_chat._build_thinking_config("gemini-3.6-flash", types)
    assert tc_36.thinking_level is not None
    assert tc_36.thinking_budget is None

    tc_30 = gemini_chat._build_thinking_config("gemini-3.0-pro-preview", types)
    assert tc_30.thinking_level is not None

    # An unparseable/unknown model name must not crash -- fall back to
    # the older, more conservative parameter rather than raising.
    tc_unknown = gemini_chat._build_thinking_config("some-future-model-name", types)
    assert tc_unknown.thinking_budget == 0


def test_token_budget_is_generous_for_the_configured_default_model(monkeypatch):
    """The visible reply must not be truncated by an overly small token
    budget, regardless of which model generation is active."""
    _clear_gemini_state()
    os.environ["GEMINI_API_KEY"] = "fake-key-for-test"

    captured_config = {}

    def fake_generate_content(model, contents, config):
        captured_config["config"] = config
        return _FakeResponse("A complete, non-truncated answer.")

    fake_client = MagicMock()
    fake_client.models.generate_content = fake_generate_content
    gemini_chat._client = fake_client

    gemini_chat.get_llm_reply("What is diabetes?")

    config = captured_config["config"]
    assert config.thinking_config is not None
    assert config.max_output_tokens >= 500
    _clear_gemini_state()


def test_mocked_off_topic_sentinel_becomes_clean_redirect(monkeypatch):
    _clear_gemini_state()
    os.environ["GEMINI_API_KEY"] = "fake-key-for-test"

    def fake_generate_content(model, contents, config):
        return _FakeResponse(gemini_chat.OFF_TOPIC_SENTINEL)

    fake_client = MagicMock()
    fake_client.models.generate_content = fake_generate_content
    gemini_chat._client = fake_client

    reply = gemini_chat.get_llm_reply("What is the capital of France?")
    assert reply is not None
    # The raw sentinel must never leak to the user -- only our own wording.
    assert gemini_chat.OFF_TOPIC_SENTINEL not in reply
    assert "only able to help" in reply.lower()
    _clear_gemini_state()


def test_mocked_api_error_falls_back_gracefully(monkeypatch):
    _clear_gemini_state()
    os.environ["GEMINI_API_KEY"] = "fake-key-for-test"

    def fake_generate_content(*a, **kw):
        raise RuntimeError("simulated network failure")

    fake_client = MagicMock()
    fake_client.models.generate_content = fake_generate_content
    gemini_chat._client = fake_client

    assert gemini_chat.get_llm_reply("I have a fever") is None
    _clear_gemini_state()
