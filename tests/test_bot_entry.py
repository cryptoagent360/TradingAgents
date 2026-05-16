"""bot.py top-level entry-point unit tests.

Covers operator-facing guards (TRADING_MODE) without actually running the
trading loop. The runner is mocked out — we only want to assert the
preflight checks fire correctly.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import bot

pytestmark = pytest.mark.unit


def test_main_refuses_live_mode_until_wired(monkeypatch, capsys):
    monkeypatch.setenv("TRADING_MODE", "live")
    with patch.object(bot, "run_paper") as run:
        rc = bot.main()
    assert rc == 2
    run.assert_not_called()


def test_main_refuses_unknown_mode(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "yolo")
    with patch.object(bot, "run_paper") as run:
        rc = bot.main()
    assert rc == 2
    run.assert_not_called()


def test_main_runs_paper_when_mode_unset(monkeypatch):
    """TRADING_MODE missing defaults to paper."""
    monkeypatch.delenv("TRADING_MODE", raising=False)
    with patch.object(bot, "run_paper", return_value=MagicMock()) as run:
        rc = bot.main()
    assert rc == 0
    run.assert_called_once()


def test_main_runs_paper_when_mode_paper(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "paper")
    with patch.object(bot, "run_paper", return_value=MagicMock()) as run:
        rc = bot.main()
    assert rc == 0
    run.assert_called_once()


def test_main_is_case_insensitive_on_mode(monkeypatch):
    """PAPER and Paper should both work — be lenient with casing."""
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    with patch.object(bot, "run_paper", return_value=MagicMock()) as run:
        rc = bot.main()
    assert rc == 0
    run.assert_called_once()
