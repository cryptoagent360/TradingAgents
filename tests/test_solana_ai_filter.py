"""Unit tests for sol_bot.ai_filter.

Verifies fail-closed behavior: any failure mode (cooldown, timeout, API
error, missing key) yields REJECT — the filter never opens a trade by
accident.
"""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import anthropic
import pytest

import sol_bot.ai_filter as ai_filter

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    """Reset cooldown clock and lazy client before each test."""
    monkeypatch.setattr(ai_filter, "_last_ai_call", 0.0)
    monkeypatch.setattr(ai_filter, "_client", None)
    yield


SAMPLE = {
    "trend": "bullish",
    "rsi": "45.0",
    "distance": "+0.30%",
    "volume": "spike",
}


def _stub_response(text: str) -> MagicMock:
    """Build a fake messages.create() response with one text block."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    return response


def test_module_import_does_not_construct_client():
    """No blocking startup — anthropic.Anthropic is built only on first call."""
    importlib.reload(ai_filter)
    assert ai_filter._client is None


def test_approve_passes_through(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _stub_response("APPROVE")
    monkeypatch.setattr(ai_filter, "_client", fake_client)
    assert ai_filter.ai_trade_filter(SAMPLE) == "APPROVE"


def test_anything_other_than_approve_becomes_reject(monkeypatch):
    """Defensive: weird text → REJECT, never accidentally APPROVE."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _stub_response("MAYBE")
    monkeypatch.setattr(ai_filter, "_client", fake_client)
    assert ai_filter.ai_trade_filter(SAMPLE) == "REJECT"


def test_cooldown_blocks_second_call(monkeypatch):
    """Two calls within AI_COOLDOWN_SECONDS — second one must REJECT without API call."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _stub_response("APPROVE")
    monkeypatch.setattr(ai_filter, "_client", fake_client)

    first = ai_filter.ai_trade_filter(SAMPLE)
    second = ai_filter.ai_trade_filter(SAMPLE)
    assert first == "APPROVE"
    assert second == "REJECT"
    assert fake_client.messages.create.call_count == 1, (
        "second call must not hit the API while cooldown is active"
    )


def test_timeout_returns_reject(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    fake_client = MagicMock()
    fake_client.messages.create.side_effect = anthropic.APITimeoutError(MagicMock())
    monkeypatch.setattr(ai_filter, "_client", fake_client)
    assert ai_filter.ai_trade_filter(SAMPLE) == "REJECT"


def test_api_error_returns_reject(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    fake_client = MagicMock()
    fake_client.messages.create.side_effect = anthropic.APIError(
        message="boom", request=MagicMock(), body=None
    )
    monkeypatch.setattr(ai_filter, "_client", fake_client)
    assert ai_filter.ai_trade_filter(SAMPLE) == "REJECT"


def test_missing_api_key_returns_reject(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(ai_filter, "_client", None)
    assert ai_filter.ai_trade_filter(SAMPLE) == "REJECT"


def test_unexpected_exception_returns_reject(monkeypatch):
    """Belt-and-suspenders: any uncategorised error still fails closed."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    fake_client = MagicMock()
    fake_client.messages.create.side_effect = ValueError("totally unexpected")
    monkeypatch.setattr(ai_filter, "_client", fake_client)
    assert ai_filter.ai_trade_filter(SAMPLE) == "REJECT"
