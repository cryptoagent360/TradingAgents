"""Open-trade lifecycle: scaling out and trailing.

The strategy splits an open long into three tranches, each with its
own exit:

* 50% closes at TP1 (entry + 1R) and the stop pulls up to breakeven
* 25% closes at TP2 (entry + 2R)
* 25% rides a chandelier-style ATR trailing stop until it hits

A bar is considered to "fill" a level when its high (for TPs) or low
(for stops) reaches that level. We pessimistically resolve same-bar
TP+stop conflicts by counting the stop first — that matches conservative
backtesting practice and the live exchange would do the same when both
orders are resting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class FillEvent:
    """A leg of a trade closing at a specific price for a specific size."""

    kind: str  # "tp1" | "tp2" | "stop" | "trail"
    price: float
    size: float


@dataclass
class OpenTrade:
    entry: float
    initial_stop: float
    size: float
    tp1_price: float
    tp2_price: float
    tp1_size: float
    tp2_size: float
    runner_size: float
    atr_at_entry: float
    trail_atr_mult: float = 1.5

    current_stop: float = field(init=False)
    remaining: float = field(init=False)
    tp1_hit: bool = field(default=False)
    tp2_hit: bool = field(default=False)
    trail_high: float = field(init=False)

    def __post_init__(self) -> None:
        self.current_stop = self.initial_stop
        self.remaining = self.size
        self.trail_high = self.entry

    @classmethod
    def open_long(
        cls,
        entry: float,
        atr_value: float,
        size: float,
        atr_mult: float,
        tp1_r: float,
        tp2_r: float,
        tp1_close_fraction: float,
        tp2_close_fraction: float,
        trail_atr_mult: float,
    ) -> "OpenTrade":
        stop = entry - atr_mult * atr_value
        r = entry - stop
        tp1_size = size * tp1_close_fraction
        tp2_size = size * tp2_close_fraction
        runner_size = size - tp1_size - tp2_size
        return cls(
            entry=entry,
            initial_stop=stop,
            size=size,
            tp1_price=entry + tp1_r * r,
            tp2_price=entry + tp2_r * r,
            tp1_size=tp1_size,
            tp2_size=tp2_size,
            runner_size=runner_size,
            atr_at_entry=atr_value,
            trail_atr_mult=trail_atr_mult,
        )

    def manage(self, *, high: float, low: float, close: float) -> List[FillEvent]:
        """Process one bar of OHLC and emit any fills.

        Order of resolution within a bar:
          1. Stop hit → close everything remaining at the stop price
          2. TP1 hit → 50% off + stop to breakeven
          3. TP2 hit → 25% off
          4. Trail update on the runner using the bar's high
        """
        events: List[FillEvent] = []

        if self.remaining <= 0:
            return events

        if low <= self.current_stop:
            events.append(
                FillEvent(
                    kind="trail" if self.tp1_hit else "stop",
                    price=self.current_stop,
                    size=self.remaining,
                )
            )
            self.remaining = 0.0
            return events

        if not self.tp1_hit and high >= self.tp1_price:
            events.append(FillEvent(kind="tp1", price=self.tp1_price, size=self.tp1_size))
            self.remaining -= self.tp1_size
            self.tp1_hit = True
            self.current_stop = max(self.current_stop, self.entry)

        if self.tp1_hit and not self.tp2_hit and high >= self.tp2_price:
            events.append(FillEvent(kind="tp2", price=self.tp2_price, size=self.tp2_size))
            self.remaining -= self.tp2_size
            self.tp2_hit = True

        if self.tp2_hit and self.remaining > 0:
            self.trail_high = max(self.trail_high, high)
            new_trail = self.trail_high - self.trail_atr_mult * self.atr_at_entry
            if new_trail > self.current_stop:
                self.current_stop = new_trail

        return events

    def realised_pnl(self, fills: List[FillEvent], fee_rate: float = 0.0) -> float:
        """PnL for the supplied fills, including a flat per-leg fee."""
        pnl = 0.0
        for ev in fills:
            gross = (ev.price - self.entry) * ev.size
            fees = (self.entry + ev.price) * ev.size * fee_rate
            pnl += gross - fees
        return pnl

    def is_closed(self) -> bool:
        return self.remaining <= 1e-12

    def to_dict(self) -> dict:
        return {
            "entry": self.entry,
            "initial_stop": self.initial_stop,
            "size": self.size,
            "tp1_price": self.tp1_price,
            "tp2_price": self.tp2_price,
            "tp1_size": self.tp1_size,
            "tp2_size": self.tp2_size,
            "runner_size": self.runner_size,
            "atr_at_entry": self.atr_at_entry,
            "trail_atr_mult": self.trail_atr_mult,
            "current_stop": self.current_stop,
            "remaining": self.remaining,
            "tp1_hit": self.tp1_hit,
            "tp2_hit": self.tp2_hit,
            "trail_high": self.trail_high,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OpenTrade":
        runtime_keys = {"current_stop", "remaining", "tp1_hit", "tp2_hit", "trail_high"}
        init_data = {k: v for k, v in data.items() if k not in runtime_keys}
        trade = cls(**init_data)
        trade.current_stop = data["current_stop"]
        trade.remaining = data["remaining"]
        trade.tp1_hit = data["tp1_hit"]
        trade.tp2_hit = data["tp2_hit"]
        trade.trail_high = data["trail_high"]
        return trade
