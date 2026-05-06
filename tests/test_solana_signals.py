"""Signal generator tests on engineered fixture data."""

import numpy as np
import pandas as pd
import pytest

from tradingagents.solana_bot.config import BotConfig
from tradingagents.solana_bot.indicators import attach_indicators
from tradingagents.solana_bot.signals import check_long_setup

pytestmark = pytest.mark.unit


def _build_uptrend_df(
    n: int = 260,
    *,
    drift: float = 0.5,
    last_close_offset: float = 0.0,
    last_open_offset: float = -0.4,
    last_volume_mult: float = 2.0,
) -> pd.DataFrame:
    """Construct a synthetic price series with a steady uptrend.

    The last bar's close lands ``last_close_offset`` above the rising
    EMA50 (so a small offset puts it inside the pullback band). Final-bar
    open/high/low and volume are set so it is a strong bullish candle on
    a volume spike unless the test overrides them.
    """
    rng = np.random.default_rng(42)
    closes = np.cumsum(np.full(n, drift) + rng.normal(0, 0.05, n)) + 100.0
    highs = closes + 0.4
    lows = closes - 0.4
    opens = closes - drift
    volumes = np.full(n, 1000.0)

    df = pd.DataFrame(
        {
            "timestamp": np.arange(n) * 3600 * 1000,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )

    enriched = attach_indicators(df, ema_fast=50, ema_slow=200, rsi_period=14, atr_period=14)
    target_close = float(enriched["ema_fast"].iloc[-1]) + last_close_offset
    target_open = target_close + last_open_offset
    target_high = target_close + 0.2
    target_low = min(target_open, target_close) - 0.2
    df.loc[df.index[-1], "open"] = target_open
    df.loc[df.index[-1], "close"] = target_close
    df.loc[df.index[-1], "high"] = target_high
    df.loc[df.index[-1], "low"] = target_low
    df.loc[df.index[-1], "volume"] = 1000.0 * last_volume_mult

    return attach_indicators(df, ema_fast=50, ema_slow=200, rsi_period=14, atr_period=14)


def _config() -> BotConfig:
    return BotConfig(rsi_long_min=0.0, rsi_long_max=100.0)  # widened for synthetic data


def test_uptrend_pullback_triggers_long():
    df = _build_uptrend_df(last_close_offset=0.0, last_volume_mult=2.0)
    sig = check_long_setup(df, _config())
    assert sig.is_long, sig.reason


def test_no_signal_when_below_ema200():
    rng = np.random.default_rng(0)
    n = 260
    closes = np.cumsum(np.full(n, -0.3) + rng.normal(0, 0.05, n)) + 200.0
    df = pd.DataFrame(
        {
            "timestamp": np.arange(n) * 3600 * 1000,
            "open": closes - 0.1,
            "high": closes + 0.2,
            "low": closes - 0.4,
            "close": closes,
            "volume": [1000.0] * n,
        }
    )
    enriched = attach_indicators(df, ema_fast=50, ema_slow=200, rsi_period=14, atr_period=14)
    sig = check_long_setup(enriched, _config())
    assert not sig.is_long
    assert "EMA200" in sig.reason or "EMA50" in sig.reason


def test_no_signal_when_pullback_too_far():
    df = _build_uptrend_df(last_close_offset=10.0)  # far above EMA50
    sig = check_long_setup(df, _config())
    assert not sig.is_long
    assert "EMA50" in sig.reason


def test_no_signal_when_rsi_outside_band():
    df = _build_uptrend_df(last_close_offset=0.0)
    cfg = BotConfig(rsi_long_min=99.0, rsi_long_max=100.0)
    sig = check_long_setup(df, cfg)
    assert not sig.is_long
    assert "RSI" in sig.reason


def test_no_signal_when_no_volume_spike():
    df = _build_uptrend_df(last_close_offset=0.0, last_volume_mult=0.5)
    sig = check_long_setup(df, _config())
    assert not sig.is_long
    assert "volume" in sig.reason


def test_no_signal_when_candle_is_bearish():
    df = _build_uptrend_df(last_close_offset=0.0, last_open_offset=0.5)  # close < open
    sig = check_long_setup(df, _config())
    assert not sig.is_long
    assert "bullish" in sig.reason


def test_no_signal_with_short_history():
    df = _build_uptrend_df(n=100)
    sig = check_long_setup(df, _config())
    assert not sig.is_long
    assert "history" in sig.reason
