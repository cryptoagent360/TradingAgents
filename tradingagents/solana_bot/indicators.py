"""Pure indicator functions over an OHLCV pandas DataFrame.

All functions take an immutable view of price data and return either a
new ``Series`` or a scalar. Computations use Wilder smoothing where
appropriate (RSI, ATR) so values match canonical TradingView output.

Expected DataFrame columns: ``open``, ``high``, ``low``, ``close``,
``volume``. Index is ignored.
"""

from __future__ import annotations

import pandas as pd


def ema(close: pd.Series, length: int) -> pd.Series:
    return close.ewm(span=length, adjust=False, min_periods=length).mean()


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    diff = close.diff()
    gain = diff.clip(lower=0.0)
    loss = -diff.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0.0, pd.NA)
    out = 100.0 - (100.0 / (1.0 + rs))
    # When avg_loss is 0 the ratio is undefined; the market was strictly up,
    # which corresponds to an RSI of 100.
    return out.fillna(100.0).where(avg_loss.notna(), other=pd.NA)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def is_bullish_candle(row: pd.Series) -> bool:
    """A close above the open, with body covering at least 50% of the range.

    The 50% rule weeds out doji-like bars where buyers and sellers are
    balanced — those are not the "strong close upward" the strategy
    requires.
    """
    open_ = float(row["open"])
    close = float(row["close"])
    high = float(row["high"])
    low = float(row["low"])
    rng = high - low
    if rng <= 0:
        return False
    body = close - open_
    return body > 0 and (body / rng) >= 0.5


def volume_spike(df: pd.DataFrame, lookback: int = 20, mult: float = 1.5) -> bool:
    """True when the latest closed bar's volume ≥ ``mult`` × the prior average.

    Uses the prior ``lookback`` bars (excluding the latest) to avoid the
    spike contaminating its own baseline.
    """
    if len(df) < lookback + 1:
        return False
    latest = float(df["volume"].iloc[-1])
    baseline = float(df["volume"].iloc[-lookback - 1 : -1].mean())
    if baseline <= 0:
        return False
    return latest >= mult * baseline


def attach_indicators(df: pd.DataFrame, *, ema_fast: int, ema_slow: int, rsi_period: int, atr_period: int) -> pd.DataFrame:
    """Return a copy of ``df`` with EMA/RSI/ATR columns attached."""
    out = df.copy()
    out["ema_fast"] = ema(out["close"], ema_fast)
    out["ema_slow"] = ema(out["close"], ema_slow)
    out["rsi"] = rsi(out["close"], rsi_period)
    out["atr"] = atr(out, atr_period)
    return out
