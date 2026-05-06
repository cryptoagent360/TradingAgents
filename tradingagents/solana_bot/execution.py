"""Execution engines.

``PaperEngine`` simulates fills against bar OHLC and is the only engine
exposed to the user in this release. ``LiveEngine`` is a stub that
raises on construction — wiring real ccxt order placement is a separate
follow-up so it can land with a focused review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from tradingagents.solana_bot.config import BotConfig
from tradingagents.solana_bot.trade import FillEvent, OpenTrade


@dataclass
class PaperFillReport:
    """Result of a single ``manage`` call."""

    fills: List[FillEvent] = field(default_factory=list)
    realised_pnl: float = 0.0


class PaperEngine:
    """Simulated execution against bar OHLC.

    Entries fill at the bar close on the bar that produced the signal.
    Subsequent management uses each later bar's high/low to resolve TP
    and stop fills (see ``OpenTrade.manage``). Fees are deducted as a
    percentage of notional on both legs (entry already implicit, exits
    via ``OpenTrade.realised_pnl``).
    """

    def __init__(self, config: BotConfig, balance: float):
        self.config = config
        self.balance = balance
        self.starting_balance = balance

    def open_long(self, *, entry_price: float, size: float, atr_value: float) -> OpenTrade:
        trade = OpenTrade.open_long(
            entry=entry_price,
            atr_value=atr_value,
            size=size,
            atr_mult=self.config.atr_mult,
            tp1_r=self.config.tp1_r,
            tp2_r=self.config.tp2_r,
            tp1_close_fraction=self.config.tp1_close_fraction,
            tp2_close_fraction=self.config.tp2_close_fraction,
            trail_atr_mult=self.config.trail_atr_mult,
        )
        self.balance -= entry_price * size * self.config.taker_fee  # entry fee only
        return trade

    def manage(self, trade: OpenTrade, *, high: float, low: float, close: float) -> PaperFillReport:
        events = trade.manage(high=high, low=low, close=close)
        if not events:
            return PaperFillReport()
        pnl = trade.realised_pnl(events, fee_rate=self.config.taker_fee)
        self.balance += pnl
        return PaperFillReport(fills=events, realised_pnl=pnl)


class LiveEngine:
    """Live exchange execution. Intentionally not implemented yet.

    The follow-up PR will:
      * Verify the API key has no withdraw permission via
        ``ccxt.fetch_account_permissions`` (where available).
      * Place market entry, then stop and TP as separate orders.
      * Reconcile partial fills against ``OpenTrade`` state.
      * Require ``--testnet`` or ``--i-have-backtested`` at the CLI.
    """

    def __init__(self, *_, **__):
        raise NotImplementedError(
            "LiveEngine is intentionally stubbed in this release. "
            "Run ``tradingagents solana backtest`` and ``tradingagents solana paper`` first; "
            "live execution lands in the next PR."
        )
