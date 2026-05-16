"""Hand-checked sanity tests for indicator functions."""

import math

import pandas as pd
import pytest

from tradingagents.solana_bot.indicators import (
    atr,
    attach_indicators,
    ema,
    is_bullish_candle,
    rsi,
    volume_spike,
)

pytestmark = pytest.mark.unit


def _series(values):
    return pd.Series(values, dtype="float64")


def test_ema_converges_to_constant_input():
    s = _series([10.0] * 50)
    out = ema(s, 5)
    assert math.isclose(out.iloc[-1], 10.0, rel_tol=1e-9)


def test_ema_first_value_after_warmup():
    # With min_periods=length, the first non-NaN value is at index length-1
    # and equals the SMA of the first window.
    s = _series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    out = ema(s, 3)
    assert pd.isna(out.iloc[0])
    assert pd.isna(out.iloc[1])
    assert not pd.isna(out.iloc[2])


def test_rsi_strictly_up_series_is_high():
    s = _series([float(i) for i in range(1, 60)])
    out = rsi(s, 14)
    last = out.iloc[-1]
    assert last >= 99.0


def test_rsi_strictly_down_series_is_low():
    s = _series([float(i) for i in range(60, 1, -1)])
    out = rsi(s, 14)
    last = out.iloc[-1]
    assert last <= 1.0


def test_atr_with_constant_range():
    # If every bar has H-L = 5 and no overnight gap, ATR should converge to 5.
    rows = []
    for i in range(60):
        rows.append({"open": 100.0, "high": 102.5, "low": 97.5, "close": 100.0})
    df = pd.DataFrame(rows)
    out = atr(df, 14)
    assert math.isclose(out.iloc[-1], 5.0, rel_tol=1e-9)


def test_is_bullish_candle_strong_close():
    row = pd.Series({"open": 100.0, "high": 110.0, "low": 99.0, "close": 109.0})
    assert is_bullish_candle(row)


def test_is_bullish_candle_doji_rejected():
    row = pd.Series({"open": 100.0, "high": 110.0, "low": 90.0, "close": 100.5})
    # Body 0.5 / range 20 = 2.5% — well below 50% threshold.
    assert not is_bullish_candle(row)


def test_is_bullish_candle_red_rejected():
    row = pd.Series({"open": 110.0, "high": 111.0, "low": 99.0, "close": 100.0})
    assert not is_bullish_candle(row)


def test_volume_spike_detects_recent_surge():
    df = pd.DataFrame({"volume": [100.0] * 20 + [200.0]})
    assert volume_spike(df, lookback=20, mult=1.5)


def test_volume_spike_rejects_quiet_bar():
    df = pd.DataFrame({"volume": [100.0] * 20 + [110.0]})
    assert not volume_spike(df, lookback=20, mult=1.5)


def test_attach_indicators_adds_columns():
    df = pd.DataFrame(
        {
            "open": [100.0] * 250,
            "high": [101.0] * 250,
            "low": [99.0] * 250,
            "close": [100.0] * 250,
            "volume": [1000.0] * 250,
        }
    )
    out = attach_indicators(df, ema_fast=50, ema_slow=200, rsi_period=14, atr_period=14)
    assert {"ema_fast", "ema_slow", "rsi", "atr"}.issubset(out.columns)
    # Last row should have all indicators populated.
    assert not out[["ema_fast", "ema_slow", "atr"]].iloc[-1].isna().any()
