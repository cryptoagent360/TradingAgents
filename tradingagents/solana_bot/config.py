"""Configuration objects for the Solana bot.

A ``BotConfig`` instance is passed through the data, signal, risk, and
execution layers. Defaults match the strategy spec in
``tradingagents/solana_bot/README.md``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

_HOME = Path(os.path.expanduser("~")) / ".tradingagents" / "solana_bot"

_SUPPORTED_TIMEFRAMES = {"1m", "5m", "15m", "30m", "1h", "4h", "1d"}


def _is_finite_positive(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def _is_finite_in_range(value: float, lo: float, hi: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and lo <= value <= hi


class BotConfigError(ValueError):
    """Raised at BotConfig construction when a field is out of range / NaN / negative."""


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

    # Live-trading safety: refuse to construct LiveEngine if the account's
    # quote-currency balance exceeds this. Forces "tiny live capital only"
    # at the start — operator must raise this constant intentionally as
    # confidence grows. Default $100. Ignored by PaperEngine.
    max_live_balance: float = 100.0

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
    def journal_path(self) -> Path:
        return self.home_dir / "journal.jsonl"

    @property
    def min_bars(self) -> int:
        # Need enough history to compute EMA200 + a stable ATR/RSI window.
        return self.ema_slow + max(self.atr_period, self.rsi_period) + 5

    def __post_init__(self) -> None:
        # Catch obviously-broken config at construction. Bad numbers silently
        # accepted here become subtle wrong behavior at trade time.
        if "/" not in self.symbol:
            raise BotConfigError(f"symbol must be 'BASE/QUOTE' form, got {self.symbol!r}")
        if self.timeframe not in _SUPPORTED_TIMEFRAMES:
            raise BotConfigError(
                f"timeframe {self.timeframe!r} not in {sorted(_SUPPORTED_TIMEFRAMES)}"
            )
        for name in ("ema_fast", "ema_slow", "rsi_period", "atr_period",
                     "volume_lookback"):
            v = getattr(self, name)
            if not (isinstance(v, int) and v > 0):
                raise BotConfigError(f"{name} must be a positive int, got {v!r}")
        if self.ema_fast >= self.ema_slow:
            raise BotConfigError(
                f"ema_fast ({self.ema_fast}) must be < ema_slow ({self.ema_slow})"
            )
        for name in ("atr_mult", "trail_atr_mult", "tp1_r", "tp2_r",
                     "volume_spike_mult", "max_live_balance"):
            v = getattr(self, name)
            if not _is_finite_positive(v):
                raise BotConfigError(f"{name} must be positive and finite, got {v!r}")
        if self.tp2_r <= self.tp1_r:
            raise BotConfigError(
                f"tp2_r ({self.tp2_r}) must be > tp1_r ({self.tp1_r})"
            )
        for name in ("risk_pct", "max_daily_loss_pct", "pullback_pct",
                     "tp1_close_fraction", "tp2_close_fraction", "taker_fee"):
            v = getattr(self, name)
            if not _is_finite_in_range(v, 0.0, 1.0):
                raise BotConfigError(f"{name} must be in [0, 1] and finite, got {v!r}")
        if self.tp1_close_fraction + self.tp2_close_fraction >= 1.0:
            raise BotConfigError(
                f"tp1_close_fraction ({self.tp1_close_fraction}) + "
                f"tp2_close_fraction ({self.tp2_close_fraction}) must sum to < 1.0 "
                f"(remainder is the trailing runner)"
            )
        if not _is_finite_in_range(self.rsi_long_min, 0.0, 100.0):
            raise BotConfigError(f"rsi_long_min must be in [0, 100], got {self.rsi_long_min!r}")
        if not _is_finite_in_range(self.rsi_long_max, 0.0, 100.0):
            raise BotConfigError(f"rsi_long_max must be in [0, 100], got {self.rsi_long_max!r}")
        if self.rsi_long_min > self.rsi_long_max:
            raise BotConfigError(
                f"rsi_long_min ({self.rsi_long_min}) must be <= rsi_long_max ({self.rsi_long_max})"
            )
