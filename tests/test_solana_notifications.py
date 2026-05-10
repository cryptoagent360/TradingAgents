"""Telegram notifier unit tests — no real network."""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock

import pytest

from tradingagents.solana_bot.notifications import TelegramNotifier

pytestmark = pytest.mark.unit


def _fake_opener(status: int = 200):
    """Build a urlopen-shaped mock that returns a context-manager response."""
    response = MagicMock()
    response.status = status
    response.__enter__ = MagicMock(return_value=response)
    response.__exit__ = MagicMock(return_value=False)
    opener = MagicMock(return_value=response)
    return opener, response


def test_notifier_is_no_op_when_token_blank():
    n = TelegramNotifier(token="", chat_id="123")
    assert not n.enabled
    assert n.notify("INFO", "hi") is False


def test_notifier_is_no_op_when_chat_id_blank():
    n = TelegramNotifier(token="abc", chat_id="")
    assert not n.enabled
    assert n.notify("INFO", "hi") is False


def test_notifier_reads_credentials_from_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "env-tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "env-chat")
    n = TelegramNotifier()
    assert n.enabled
    assert n.token == "env-tok"
    assert n.chat_id == "env-chat"


def test_notify_posts_to_telegram_with_correct_body():
    opener, _ = _fake_opener(status=200)
    n = TelegramNotifier(token="t", chat_id="c", opener=opener)
    assert n.notify("WARN", "kill switch tripped") is True

    assert opener.call_count == 1
    request = opener.call_args.args[0]
    assert request.full_url == "https://api.telegram.org/bott/sendMessage"
    assert request.method == "POST"
    payload = json.loads(request.data.decode("utf-8"))
    assert payload == {"chat_id": "c", "text": "[WARN] kill switch tripped"}


def test_notify_returns_false_on_non_200():
    opener, _ = _fake_opener(status=500)
    n = TelegramNotifier(token="t", chat_id="c", opener=opener)
    assert n.notify("INFO", "hello") is False


def test_notify_swallows_url_errors():
    """Network failure must never raise out of notify()."""
    def boom(*_args, **_kwargs):
        raise urllib.error.URLError("DNS fail")

    n = TelegramNotifier(token="t", chat_id="c", opener=boom)
    # The whole point is that this does not raise.
    assert n.notify("ERROR", "drift detected") is False


def test_notify_swallows_unexpected_exceptions():
    """Any uncategorised error still returns False, never raises."""
    def boom(*_args, **_kwargs):
        raise RuntimeError("totally unexpected")

    n = TelegramNotifier(token="t", chat_id="c", opener=boom)
    assert n.notify("ERROR", "boom") is False


def test_notify_strips_whitespace_from_creds():
    """Trailing newlines (common when sourcing from .env) must not break the URL."""
    opener, _ = _fake_opener(status=200)
    n = TelegramNotifier(token="  abc\n", chat_id=" 999 ", opener=opener)
    assert n.notify("INFO", "ok") is True
    request = opener.call_args.args[0]
    assert request.full_url == "https://api.telegram.org/botabc/sendMessage"
