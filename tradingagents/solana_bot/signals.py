"""Long-setup signal generator.

The bot enters LONG when a strict five-of-five gate passes on the most
recently closed bar:

1. Price > EMA200 (macro bullish)
2. EMA50 > EMA200 (trend confirmed)
3. ``|close − EMA50| / EMA50 < pullback_pct`` (price has retraced to value)
4. RSI ∈ [rsi_long_min, rsi_long_max] (momentum cooled but not bearish)
5. The bar itself is a strong bullish candle on a volume spike

Short setups are deferred to Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tradingagents.solana_bot.config import BotConfig
from tradingagents.solana_bot.indicators import is_bullish_candle, volume_spike


@dataclass(frozen=True)
class Signal:
    direction: str  # "LONG" | "NO_TRADE"
    reason: str

    @property
    def is_long(self) -> bool:
        return self.direction == "LONG"


NO_TRADE = lambda reason: Signal("NO_TRADE", reason)  # noqa: E731


def check_long_setup(df: pd.DataFrame, config: BotConfig) -> Signal:
    """Inspect the latest closed bar and return a Signal.

    ``df`` must already have ``ema_fast``, ``ema_slow``, ``rsi``, ``atr``
    columns attached (via ``indicators.attach_indicators``). The latest
    row is treated as the just-closed bar.
    """
    if len(df) < config.min_bars:
        return NO_TRADE(f"insufficient history: have {len(df)}, need {config.min_bars}")

    last = df.iloc[-1]
    close = float(last["close"])
    ema_fast = float(last["ema_fast"])
    ema_slow = float(last["ema_slow"])
    rsi_value = last["rsi"]
    atr_value = last["atr"]

    if pd.isna(ema_fast) or pd.isna(ema_slow) or pd.isna(rsi_value) or pd.isna(atr_value):
        return NO_TRADE("indicators not yet warm")

    rsi_value = float(rsi_value)
    atr_value = float(atr_value)

    if close <= ema_slow:
        return NO_TRADE("price below EMA200")
    if ema_fast <= ema_slow:
        return NO_TRADE("EMA50 not above EMA200")

    pullback_distance = abs(close - ema_fast) / ema_fast
    if pullback_distance >= config.pullback_pct:
        return NO_TRADE(f"price not within {config.pullback_pct:.0%} of EMA50 (dist={pullback_distance:.3f})")

    if not (config.rsi_long_min <= rsi_value <= config.rsi_long_max):
        return NO_TRADE(f"RSI {rsi_value:.1f} outside [{config.rsi_long_min}, {config.rsi_long_max}]")

    if not is_bullish_candle(last):
        return NO_TRADE("no bullish confirmation candle")

    if not volume_spike(df, config.volume_lookback, config.volume_spike_mult):
        return NO_TRADE("no volume spike")

    if atr_value <= 0:
        return NO_TRADE("ATR is zero")

    return Signal("LONG", "all five conditions satisfied")
