"""bot.py top-level entry-point unit tests.

Covers operator-facing guards (TRADING_MODE, EXECUTE_TRADES, KRAKEN_*
presence, confirm_trade_only_key) without actually running the trading
loop. The runner and the engine factory are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import bot
from sol_bot import ai_filter as ai_filter_module

pytestmark = pytest.mark.unit


# ------------------------ Mode preflight ----------------------------------


def test_main_refuses_unknown_mode(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "yolo")
    with patch.object(bot, "run_paper") as run:
        rc = bot.main()
    assert rc == 2
    run.assert_not_called()


def test_main_runs_paper_when_mode_unset(monkeypatch):
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
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    with patch.object(bot, "run_paper", return_value=MagicMock()) as run:
        rc = bot.main()
    assert rc == 0
    run.assert_called_once()


# ------------------------ Live-mode preflight ------------------------------


def test_main_refuses_live_when_execute_trades_is_false(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("KRAKEN_API_KEY", "k")
    monkeypatch.setenv("KRAKEN_API_SECRET", "s")
    monkeypatch.setenv("CONFIRM_TRADE_ONLY_KEY", "yes")
    # The shipped default has EXECUTE_TRADES = False — verify the guard fires.
    monkeypatch.setattr(ai_filter_module, "EXECUTE_TRADES", False)
    monkeypatch.setattr(bot, "EXECUTE_TRADES", False)
    with patch.object(bot, "run_paper") as run:
        rc = bot.main()
    assert rc == 2
    run.assert_not_called()


def test_main_refuses_live_when_kraken_creds_missing(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setattr(bot, "EXECUTE_TRADES", True)
    monkeypatch.delenv("KRAKEN_API_KEY", raising=False)
    monkeypatch.delenv("KRAKEN_API_SECRET", raising=False)
    with patch.object(bot, "run_paper") as run, pytest.raises(SystemExit) as exc:
        bot.main()
    assert exc.value.code == 2
    run.assert_not_called()


def test_main_refuses_live_when_trade_only_not_acknowledged(monkeypatch):
    """confirm_trade_only_key=False (the BotConfig default) blocks live."""
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setattr(bot, "EXECUTE_TRADES", True)
    monkeypatch.setenv("KRAKEN_API_KEY", "k")
    monkeypatch.setenv("KRAKEN_API_SECRET", "s")
    monkeypatch.delenv("CONFIRM_TRADE_ONLY_KEY", raising=False)
    with patch.object(bot, "run_paper") as run, pytest.raises(SystemExit) as exc:
        bot.main()
    assert exc.value.code == 2
    run.assert_not_called()


def test_main_builds_live_engine_when_all_gates_pass(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setattr(bot, "EXECUTE_TRADES", True)
    monkeypatch.setenv("KRAKEN_API_KEY", "k")
    monkeypatch.setenv("KRAKEN_API_SECRET", "s")
    monkeypatch.setenv("CONFIRM_TRADE_ONLY_KEY", "yes")
    monkeypatch.setenv("BOT_SYMBOL", "SOL/USD")

    fake_engine = MagicMock(name="LiveEngine")
    with patch.object(bot, "LiveEngine", return_value=fake_engine) as live_ctor, \
         patch.object(bot, "run_paper", return_value=MagicMock()) as run:
        rc = bot.main()
    assert rc == 0
    live_ctor.assert_called_once()
    # The engine instance must be passed into run_paper
    assert run.call_args.kwargs.get("engine") is fake_engine
    # stop_requested predicate is also wired
    assert callable(run.call_args.kwargs.get("stop_requested"))


# ------------------------ SIGTERM handler ----------------------------------


def test_sigterm_handler_flips_stop_flag(monkeypatch):
    """Sending SIGTERM (simulated by calling the installed handler) sets
    the module-level _STOP_REQUESTED flag the runner observes."""
    monkeypatch.setattr(bot, "_STOP_REQUESTED", False)
    log = MagicMock()
    bot._install_sigterm_handler(log)
    import signal as _signal
    handler = _signal.getsignal(_signal.SIGTERM)
    handler(_signal.SIGTERM, None)
    assert bot._STOP_REQUESTED is True
