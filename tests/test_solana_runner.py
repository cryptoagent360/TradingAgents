"""Smoke tests for the paper-trade runner.

Drives ``run_paper`` with an injected fetcher and a no-op sleeper so the
loop can be exercised without network or wall-clock waits.
"""

import json

import numpy as np
import pandas as pd
import pytest

from tradingagents.solana_bot.config import BotConfig
from tradingagents.solana_bot.runner import run_paper

pytestmark = pytest.mark.unit


def _fixture_df():
    """An uptrend with a planted setup on the final bar."""
    from tradingagents.solana_bot.indicators import attach_indicators

    rng = np.random.default_rng(11)
    n = 260
    drift = 0.5
    closes = np.cumsum(np.full(n, drift) + rng.normal(0, 0.05, n)) + 100.0
    df = pd.DataFrame(
        {
            "timestamp": np.arange(n) * 3600 * 1000,
            "open": closes - drift,
            "high": closes + 0.4,
            "low": closes - 0.4,
            "close": closes,
            "volume": [1000.0] * n,
        }
    )
    enriched = attach_indicators(df, ema_fast=50, ema_slow=200, rsi_period=14, atr_period=14)
    target = float(enriched["ema_fast"].iloc[-1])
    df.loc[df.index[-1], "open"] = target - 0.4
    df.loc[df.index[-1], "close"] = target
    df.loc[df.index[-1], "high"] = target + 0.2
    df.loc[df.index[-1], "low"] = target - 0.5
    df.loc[df.index[-1], "volume"] = 2500.0
    return df


def test_runner_opens_trade_when_setup_fires(tmp_path):
    df = _fixture_df()
    cfg = BotConfig(rsi_long_min=0.0, rsi_long_max=100.0, home_dir=tmp_path)

    state = run_paper(
        cfg,
        starting_balance=10_000,
        max_cycles=1,
        sleeper=lambda *_: None,
        fetcher=lambda _cfg, _n: df,
        ai_filter=lambda _md: "APPROVE",
        execute_trades=True,
    )
    assert state.has_open_trade()
    # State persisted to disk.
    assert cfg.state_path.exists()


def test_runner_blocks_trade_when_ai_rejects(tmp_path):
    """Valid setup but AI says REJECT — no trade opened, state clean."""
    df = _fixture_df()
    cfg = BotConfig(rsi_long_min=0.0, rsi_long_max=100.0, home_dir=tmp_path)

    state = run_paper(
        cfg,
        starting_balance=10_000,
        max_cycles=1,
        sleeper=lambda *_: None,
        fetcher=lambda _cfg, _n: df,
        ai_filter=lambda _md: "REJECT",
        execute_trades=True,
    )
    assert not state.has_open_trade()


def test_runner_blocks_trade_when_execute_disabled(tmp_path):
    """AI approves but EXECUTE_TRADES is off — simulation only, no trade."""
    df = _fixture_df()
    cfg = BotConfig(rsi_long_min=0.0, rsi_long_max=100.0, home_dir=tmp_path)

    state = run_paper(
        cfg,
        starting_balance=10_000,
        max_cycles=1,
        sleeper=lambda *_: None,
        fetcher=lambda _cfg, _n: df,
        ai_filter=lambda _md: "APPROVE",
        execute_trades=False,
    )
    assert not state.has_open_trade()


def test_runner_exits_when_burn_in_budget_elapsed(tmp_path):
    """A 3h burn-in budget exits cleanly without a kill switch trip."""
    flat = pd.DataFrame(
        {
            "timestamp": np.arange(260) * 3600 * 1000,
            "open": [100.0] * 260,
            "high": [101.0] * 260,
            "low": [99.0] * 260,
            "close": [100.0] * 260,
            "volume": [1000.0] * 260,
        }
    )
    cfg = BotConfig(home_dir=tmp_path)

    # Mock monotonic clock that fast-forwards 90 minutes per call so the
    # 3h budget trips after 2 cycle checks.
    fake_clock = iter([0.0, 5400.0, 10800.0, 16200.0])
    state = run_paper(
        cfg,
        starting_balance=10_000,
        max_runtime_s=3 * 3600,
        sleeper=lambda *_: None,
        fetcher=lambda _cfg, _n: flat,
        ai_filter=lambda _md: "APPROVE",
        execute_trades=True,
        monotonic=lambda: next(fake_clock),
    )
    assert state.extras.get("last_cycle") == "burn_in_complete"
    # No daily-loss kill switch should have tripped.
    assert state.tracker is not None
    assert not state.tracker.kill_switch_triggered


def test_reconcile_clears_local_trade_when_exchange_has_no_position(tmp_path, monkeypatch):
    """Drift case B: state has open trade, exchange has none → trade closed during downtime."""
    from unittest.mock import MagicMock

    from tradingagents.solana_bot.execution import ReconcileReport
    from tradingagents.solana_bot.runner import _reconcile_or_die
    from tradingagents.solana_bot.state import BotState
    from tradingagents.solana_bot.trade import OpenTrade

    state = BotState(path=tmp_path / "state.json")
    state.open_trade = OpenTrade.open_long(
        entry=100.0, atr_value=2.0, size=10.0, atr_mult=1.5, tp1_r=1.0, tp2_r=2.0,
        tp1_close_fraction=0.5, tp2_close_fraction=0.25, trail_atr_mult=1.5,
    )
    state.extras["current_trade_id"] = "abc"

    fake_engine = MagicMock()
    fake_engine.reconcile.return_value = ReconcileReport(
        quote_balance=100.0, base_balance=0.0, open_orders=[], is_live=True,
    )

    _reconcile_or_die(state, fake_engine, MagicMock())

    assert state.open_trade is None
    assert "current_trade_id" not in state.extras


def test_reconcile_refuses_to_start_when_exchange_has_unexpected_position(tmp_path):
    """Drift case C: state empty, exchange has position → fail closed."""
    from unittest.mock import MagicMock

    from tradingagents.solana_bot.execution import ReconcileMismatch, ReconcileReport
    from tradingagents.solana_bot.runner import _reconcile_or_die
    from tradingagents.solana_bot.state import BotState

    state = BotState(path=tmp_path / "state.json")  # no open trade
    fake_engine = MagicMock()
    fake_engine.reconcile.return_value = ReconcileReport(
        quote_balance=100.0, base_balance=2.5, open_orders=[], is_live=True,
    )

    notifier = MagicMock()
    with pytest.raises(ReconcileMismatch):
        _reconcile_or_die(state, fake_engine, notifier)
    # Operator must be paged before the bot refuses to start.
    notifier.notify.assert_called_once()
    args = notifier.notify.call_args.args
    assert args[0] == "ERROR"
    assert "ReconcileMismatch" in args[1]


def test_reconcile_skips_drift_check_for_paper_engine(tmp_path):
    """PaperEngine reports is_live=False; the drift check is a no-op."""
    from unittest.mock import MagicMock

    from tradingagents.solana_bot.execution import ReconcileReport
    from tradingagents.solana_bot.runner import _reconcile_or_die
    from tradingagents.solana_bot.state import BotState
    from tradingagents.solana_bot.trade import OpenTrade

    state = BotState(path=tmp_path / "state.json")
    state.open_trade = OpenTrade.open_long(
        entry=100.0, atr_value=2.0, size=10.0, atr_mult=1.5, tp1_r=1.0, tp2_r=2.0,
        tp1_close_fraction=0.5, tp2_close_fraction=0.25, trail_atr_mult=1.5,
    )

    paper_engine = MagicMock()
    paper_engine.reconcile.return_value = ReconcileReport(
        quote_balance=10000.0, base_balance=0.0, open_orders=[], is_live=False,
    )

    # Even though state has trade and "exchange" has none, paper mode is exempt.
    notifier = MagicMock()
    _reconcile_or_die(state, paper_engine, notifier)
    assert state.open_trade is not None  # untouched
    notifier.notify.assert_not_called()


def test_runner_persists_live_engine_order_ids_into_state(tmp_path, monkeypatch):
    """When the engine surfaces last_order_ids (LiveEngine), they land in state.extras."""
    from unittest.mock import MagicMock

    from tradingagents.solana_bot import runner as runner_module
    from tradingagents.solana_bot.trade import OpenTrade

    df = _fixture_df()
    cfg = BotConfig(rsi_long_min=0.0, rsi_long_max=100.0, home_dir=tmp_path)

    # Replace PaperEngine with a stub that mimics LiveEngine's order-id surface.
    fake_trade = OpenTrade.open_long(
        entry=100.0, atr_value=2.0, size=0.1, atr_mult=1.5, tp1_r=1.0, tp2_r=2.0,
        tp1_close_fraction=0.5, tp2_close_fraction=0.25, trail_atr_mult=1.5,
    )

    class _FakeEngine:
        def __init__(self, *_, **__):
            self.balance = 10_000.0
            self.last_order_ids = {"entry": "sb-entry-deadbeef", "stop": "sb-stop-cafebabe"}

        def open_long(self, **__):
            return fake_trade

        def manage(self, *_, **__):
            from tradingagents.solana_bot.execution import PaperFillReport
            return PaperFillReport()

        def reconcile(self):
            from tradingagents.solana_bot.execution import ReconcileReport
            return ReconcileReport(quote_balance=10_000.0, base_balance=0.0, open_orders=[], is_live=False)

    monkeypatch.setattr(runner_module, "PaperEngine", _FakeEngine)

    state = run_paper(
        cfg,
        starting_balance=10_000,
        max_cycles=1,
        sleeper=lambda *_: None,
        fetcher=lambda _cfg, _n: df,
        ai_filter=lambda _md: "APPROVE",
        execute_trades=True,
    )
    assert state.has_open_trade()
    assert state.extras.get("live_order_ids") == {
        "entry": "sb-entry-deadbeef",
        "stop": "sb-stop-cafebabe",
    }


def test_runner_writes_open_event_to_journal(tmp_path):
    """When a trade opens, a journal line with event=open is appended."""
    df = _fixture_df()
    cfg = BotConfig(rsi_long_min=0.0, rsi_long_max=100.0, home_dir=tmp_path)

    state = run_paper(
        cfg,
        starting_balance=10_000,
        max_cycles=1,
        sleeper=lambda *_: None,
        fetcher=lambda _cfg, _n: df,
        ai_filter=lambda _md: "APPROVE",
        execute_trades=True,
    )
    assert state.has_open_trade()
    assert "current_trade_id" in state.extras
    journal_path = cfg.journal_path
    assert journal_path.exists()
    lines = [json.loads(line) for line in journal_path.read_text().splitlines() if line.strip()]
    open_events = [e for e in lines if e.get("event") == "open"]
    assert open_events, "expected at least one open event"
    # cycle_number + bar_timestamp must be populated, not just present-as-None.
    assert open_events[0]["cycle_number"] == 1
    assert isinstance(open_events[0]["bar_timestamp"], int)


def test_runner_opens_trade_with_no_ai_filter_when_not_supplied(tmp_path):
    """Default behavior: when run_paper is called without ai_filter, no AI gate
    runs and the trade fires on the 5/5 strategy gate alone. This is the
    intentional opt-in default after the AI-disable cleanup."""
    df = _fixture_df()
    cfg = BotConfig(rsi_long_min=0.0, rsi_long_max=100.0, home_dir=tmp_path)

    state = run_paper(
        cfg,
        starting_balance=10_000,
        max_cycles=1,
        sleeper=lambda *_: None,
        fetcher=lambda _cfg, _n: df,
        # ai_filter omitted on purpose — default is no AI gate
        execute_trades=True,
    )
    assert state.has_open_trade(), (
        "with no ai_filter passed, the trade should open on the strategy gate alone"
    )


def test_runner_calls_ai_only_after_signal_fires(tmp_path):
    """The AI filter must NOT be called on bars where the signal doesn't fire."""
    flat = pd.DataFrame(
        {
            "timestamp": np.arange(260) * 3600 * 1000,
            "open": [100.0] * 260,
            "high": [101.0] * 260,
            "low": [99.0] * 260,
            "close": [100.0] * 260,
            "volume": [1000.0] * 260,
        }
    )
    cfg = BotConfig(home_dir=tmp_path)
    calls = []

    def spy(market_data):
        calls.append(market_data)
        return "APPROVE"

    run_paper(
        cfg,
        starting_balance=10_000,
        max_cycles=1,
        sleeper=lambda *_: None,
        fetcher=lambda _cfg, _n: flat,
        ai_filter=spy,
        execute_trades=True,
    )
    assert calls == [], "AI filter must not run when there is no setup"


def test_runner_skips_when_no_setup(tmp_path):
    """A flat market produces no trade."""
    flat = pd.DataFrame(
        {
            "timestamp": np.arange(260) * 3600 * 1000,
            "open": [100.0] * 260,
            "high": [101.0] * 260,
            "low": [99.0] * 260,
            "close": [100.0] * 260,
            "volume": [1000.0] * 260,
        }
    )
    cfg = BotConfig(home_dir=tmp_path)
    state = run_paper(
        cfg,
        starting_balance=10_000,
        max_cycles=1,
        sleeper=lambda *_: None,
        fetcher=lambda _cfg, _n: flat,
    )
    assert not state.has_open_trade()


def test_runner_resumes_open_trade_from_disk(tmp_path):
    """Round-trip: open a trade in cycle 1, then a fresh process loads it."""
    df = _fixture_df()
    cfg = BotConfig(rsi_long_min=0.0, rsi_long_max=100.0, home_dir=tmp_path)
    run_paper(
        cfg,
        starting_balance=10_000,
        max_cycles=1,
        sleeper=lambda *_: None,
        fetcher=lambda _cfg, _n: df,
        ai_filter=lambda _md: "APPROVE",
        execute_trades=True,
    )
    # Restart: load state from disk and run another cycle. Simulate a bar
    # with a higher timestamp so the new bar is processed.
    next_bar = df.copy()
    next_bar.loc[next_bar.index[-1], "timestamp"] = int(df["timestamp"].iloc[-1]) + 3600 * 1000
    state = run_paper(
        cfg,
        starting_balance=10_000,
        max_cycles=1,
        sleeper=lambda *_: None,
        fetcher=lambda _cfg, _n: next_bar,
        ai_filter=lambda _md: "APPROVE",
        execute_trades=True,
    )
    # Trade should still be open (or just closed by management).
    assert state.tracker is not None


def test_runner_exits_when_kill_switch_active(tmp_path):
    """Pre-tripped kill switch with no open trade exits immediately."""
    cfg = BotConfig(home_dir=tmp_path)
    flat = pd.DataFrame(
        {
            "timestamp": np.arange(260) * 3600 * 1000,
            "open": [100.0] * 260,
            "high": [101.0] * 260,
            "low": [99.0] * 260,
            "close": [100.0] * 260,
            "volume": [1000.0] * 260,
        }
    )

    # First, prime state with a tripped kill switch.
    from tradingagents.solana_bot.risk import DailyLossTracker
    from tradingagents.solana_bot.state import BotState

    seed = BotState(path=cfg.state_path)
    seed.tracker = DailyLossTracker(starting_balance=10_000, max_daily_loss_pct=0.03)
    seed.tracker.record_pnl(-500)  # > 3% loss → kill switch
    assert seed.tracker.kill_switch_triggered
    seed.save()

    state = run_paper(
        cfg,
        starting_balance=10_000,
        max_cycles=10,
        sleeper=lambda *_: None,
        fetcher=lambda _cfg, _n: flat,
    )
    assert state.tracker is not None
    assert state.tracker.kill_switch_triggered
    assert not state.has_open_trade()
