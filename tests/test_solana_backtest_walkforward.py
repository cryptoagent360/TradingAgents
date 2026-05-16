"""Walk-forward backtest validation across multiple market regimes.

The strategy's parameters are hardcoded in ``BotConfig``, so 'walk-forward'
here means: does it hold up across distinct regimes when run on consecutive
non-overlapping windows? Expectations are bounded sanity checks, not
assertions of excellence:

* Most regimes produce *some* trades (the 5/5 gate is strict but not lethal
  in a trending or recovering market)
* No single regime catastrophically blows up the account (per-window max
  drawdown < 35% is the alarm threshold)
* Aggregate equity across all regimes does not drop > 30% from start

A strategy that fails any of these is one nobody should be deploying with
real money — these tests guard that floor, not the ceiling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.solana_bot.backtest import run_backtest
from tradingagents.solana_bot.config import BotConfig
from tradingagents.solana_bot.indicators import attach_indicators

pytestmark = pytest.mark.unit


def _regime(seed: int, n: int, drift: float, noise: float, setups_every: int = 100) -> pd.DataFrame:
    """Generate ``n`` hourly bars with given drift/noise + engineered pullback setups.

    A pullback setup is planted every ``setups_every`` bars after warmup: the
    close is pulled to the running EMA50 and the bar is turned into a strong
    bullish candle on a volume spike. This gives the strategy realistic chances
    to fire across the window without depending on lucky random walks.
    """
    rng = np.random.default_rng(seed)
    base = 100.0
    closes = np.cumsum(np.full(n, drift) + rng.normal(0, noise, n)) + base
    # Prevent the price from going negative on long downtrends — clip at 1.0.
    closes = np.maximum(closes, 1.0)

    df = pd.DataFrame({
        "timestamp": np.arange(n) * 3600 * 1000,
        "open": closes - drift,
        "high": closes + abs(drift) + 0.3,
        "low": closes - abs(drift) - 0.3,
        "close": closes,
        "volume": [1000.0] * n,
    })

    # Compute indicators once and engineer a pullback bar every ``setups_every``
    # bars (after the EMA200 warmup window). The pullback bar pulls close to
    # EMA50, makes it a strong bullish candle, and spikes volume — but only
    # IF the macro trend is favourable (price > EMA200 and EMA50 > EMA200).
    enriched = attach_indicators(df, ema_fast=50, ema_slow=200, rsi_period=14, atr_period=14)
    for i in range(260, n, setups_every):
        ema_fast = enriched["ema_fast"].iloc[i]
        ema_slow = enriched["ema_slow"].iloc[i]
        close = enriched["close"].iloc[i]
        # Only plant the setup if the macro trend allows it; otherwise the bar
        # would fail the strategy gate anyway and we waste the slot.
        if pd.isna(ema_fast) or pd.isna(ema_slow) or close <= ema_slow or ema_fast <= ema_slow:
            continue
        target = float(ema_fast)
        df.loc[i, "open"] = target - 0.4
        df.loc[i, "close"] = target
        df.loc[i, "high"] = target + 0.2
        df.loc[i, "low"] = target - 0.5
        df.loc[i, "volume"] = 2500.0

    return df


def _max_drawdown(equity: pd.DataFrame) -> float:
    if equity.empty:
        return 0.0
    running_peak = equity["equity"].cummax()
    drawdown = (equity["equity"] - running_peak) / running_peak
    return abs(float(drawdown.min()))


REGIMES = [
    # (label, seed, n_bars, drift, noise)
    ("strong_uptrend", 1, 500, 0.40, 0.10),
    ("mild_uptrend",   2, 500, 0.15, 0.20),
    ("choppy",         3, 500, 0.05, 0.40),
    ("recovery",       4, 500, 0.25, 0.30),
]


def test_walkforward_no_single_regime_catastrophically_loses():
    """Per-window max drawdown stays under 35% — the alarm threshold."""
    cfg = BotConfig(rsi_long_min=0.0, rsi_long_max=100.0)
    failures = []
    for label, seed, n, drift, noise in REGIMES:
        df = _regime(seed, n, drift, noise)
        result = run_backtest(df, cfg, starting_balance=10_000)
        dd = _max_drawdown(result.equity_curve)
        if dd > 0.35:
            failures.append(f"{label}: max_drawdown={dd:.1%}")
    assert not failures, f"regimes blew the 35% drawdown floor: {failures}"


def test_walkforward_takes_trades_in_favorable_regimes():
    """In trending regimes the strategy must find *some* setups."""
    cfg = BotConfig(rsi_long_min=0.0, rsi_long_max=100.0)
    trend_regimes = [r for r in REGIMES if r[0] in ("strong_uptrend", "mild_uptrend", "recovery")]
    zero_trade_regimes = []
    for label, seed, n, drift, noise in trend_regimes:
        df = _regime(seed, n, drift, noise)
        result = run_backtest(df, cfg, starting_balance=10_000)
        if result.summary["trades"] == 0:
            zero_trade_regimes.append(label)
    assert not zero_trade_regimes, (
        f"strategy took 0 trades in favourable regimes: {zero_trade_regimes} — "
        f"the 5/5 gate may be over-tuned"
    )


def test_walkforward_aggregate_equity_does_not_collapse():
    """Summed across all regimes, equity must not drop > 30% from start."""
    cfg = BotConfig(rsi_long_min=0.0, rsi_long_max=100.0)
    starting = 10_000.0
    total_pnl = 0.0
    for _label, seed, n, drift, noise in REGIMES:
        df = _regime(seed, n, drift, noise)
        result = run_backtest(df, cfg, starting_balance=starting)
        total_pnl += result.summary["total_pnl"]
    final_equity = starting + total_pnl
    assert final_equity > starting * 0.70, (
        f"aggregate equity collapsed: started {starting}, ended {final_equity:.2f}"
    )


def test_walkforward_aggregate_win_rate_is_sane():
    """Trend-following expectation: aggregate win rate > 25% across all regimes."""
    cfg = BotConfig(rsi_long_min=0.0, rsi_long_max=100.0)
    total_trades = 0
    total_wins = 0
    for _label, seed, n, drift, noise in REGIMES:
        df = _regime(seed, n, drift, noise)
        result = run_backtest(df, cfg, starting_balance=10_000)
        total_trades += result.summary["trades"]
        total_wins += result.summary["wins"]
    if total_trades == 0:
        pytest.skip("no trades across any regime — covered by the takes_trades test")
    win_rate = total_wins / total_trades
    assert win_rate >= 0.25, (
        f"aggregate win rate {win_rate:.1%} is below 25% across "
        f"{total_trades} trades — strategy edge looks broken"
    )
