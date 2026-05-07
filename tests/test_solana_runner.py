"""Smoke tests for the paper-trade runner.

Drives ``run_paper`` with an injected fetcher and a no-op sleeper so the
loop can be exercised without network or wall-clock waits.
"""

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
