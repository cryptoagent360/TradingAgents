"""Configuration objects for the Solana bot.

A ``BotConfig`` instance is passed through the data, signal, risk, and
execution layers. Defaults match the strategy spec in
``tradingagents/solana_bot/README.md``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_HOME = Path(os.path.expanduser("~")) / ".tradingagents" / "solana_bot"


@dataclass(frozen=True)
class BotConfig:
    symbol: str = "SOL/USDT"
    timeframe: str = "1h"
    exchange: str = "binance"

    ema_fast: int = 50
    ema_slow: int = 200
    rsi_period: int = 14
    rsi_long_min: float = 40.0
    rsi_long_max: float = 55.0
    pullback_pct: float = 0.02
    volume_lookback: int = 20
    volume_spike_mult: float = 1.5
    atr_period: int = 14
    atr_mult: float = 1.5

    risk_pct: float = 0.01
    max_daily_loss_pct: float = 0.03
    tp1_r: float = 1.0
    tp2_r: float = 2.0
    tp1_close_fraction: float = 0.5
    tp2_close_fraction: float = 0.25
    trail_atr_mult: float = 1.5

    taker_fee: float = 0.001

    home_dir: Path = field(default_factory=lambda: _HOME)

    @property
    def cache_dir(self) -> Path:
        return self.home_dir / "cache"

    @property
    def state_path(self) -> Path:
        return self.home_dir / "state.json"

    @property
    def backtests_dir(self) -> Path:
        return self.home_dir / "backtests"

    @property
    def min_bars(self) -> int:
        # Need enough history to compute EMA200 + a stable ATR/RSI window.
        return self.ema_slow + max(self.atr_period, self.rsi_period) + 5
