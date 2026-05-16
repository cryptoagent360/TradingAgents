"""End-to-end backtester test on engineered synthetic OHLCV."""

import numpy as np
import pandas as pd
import pytest

from tradingagents.solana_bot.backtest import run_backtest
from tradingagents.solana_bot.config import BotConfig
from tradingagents.solana_bot.indicators import attach_indicators

pytestmark = pytest.mark.unit


def _build_signal_then_runup(n_warmup=260, n_followup=10):
    """Steady uptrend, an engineered pullback bar, then more uptrend."""
    rng = np.random.default_rng(7)
    drift = 0.5
    closes = np.cumsum(np.full(n_warmup, drift) + rng.normal(0, 0.05, n_warmup)) + 100.0
    df = pd.DataFrame(
        {
            "open": closes - drift,
            "high": closes + 0.4,
            "low": closes - 0.4,
            "close": closes,
            "volume": [1000.0] * n_warmup,
        }
    )

    # Configure the last warmup bar so check_long_setup fires on it: pull the
    # close down to the EMA50 and make it a strong bullish candle on volume.
    enriched = attach_indicators(df, ema_fast=50, ema_slow=200, rsi_period=14, atr_period=14)
    target = float(enriched["ema_fast"].iloc[-1])
    df.loc[df.index[-1], "open"] = target - 0.4
    df.loc[df.index[-1], "close"] = target
    df.loc[df.index[-1], "high"] = target + 0.2
    df.loc[df.index[-1], "low"] = target - 0.5
    df.loc[df.index[-1], "volume"] = 2500.0

    # Follow-up: strong runup so price hits both TPs and trails.
    last_close = float(df["close"].iloc[-1])
    extra = []
    px = last_close
    for _ in range(n_followup):
        px += 1.5
        extra.append(
            {
                "open": px - 1.0,
                "high": px + 0.5,
                "low": px - 1.2,
                "close": px,
                "volume": 1200.0,
            }
        )
    extra_df = pd.DataFrame(extra)
    full = pd.concat([df, extra_df], ignore_index=True)
    full.insert(0, "timestamp", np.arange(len(full)) * 3600 * 1000)
    return full


def test_backtest_takes_engineered_long_and_makes_money():
    df = _build_signal_then_runup()
    cfg = BotConfig(rsi_long_min=0.0, rsi_long_max=100.0)  # widened for synthetic data
    result = run_backtest(df, cfg, starting_balance=10_000)
    assert result.summary["trades"] >= 1
    assert result.summary["total_pnl"] > 0
    # Equity ends above starting balance.
    assert result.summary["ending_balance"] > 10_000


def test_backtest_writes_outputs(tmp_path):
    df = _build_signal_then_runup()
    cfg = BotConfig(rsi_long_min=0.0, rsi_long_max=100.0)
    out_dir = tmp_path / "bt"
    run_backtest(df, cfg, starting_balance=10_000, output_dir=out_dir)
    assert (out_dir / "trades.csv").exists()
    assert (out_dir / "equity.csv").exists()
    assert (out_dir / "summary.json").exists()


def test_backtest_rejects_too_short_history():
    cfg = BotConfig()
    short = pd.DataFrame(
        {
            "timestamp": np.arange(50) * 3600 * 1000,
            "open": [100.0] * 50,
            "high": [101.0] * 50,
            "low": [99.0] * 50,
            "close": [100.0] * 50,
            "volume": [1000.0] * 50,
        }
    )
    with pytest.raises(ValueError):
        run_backtest(short, cfg, starting_balance=10_000)
