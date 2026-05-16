"""Execution engines.

``PaperEngine`` simulates fills against bar OHLC and is the engine the
runner uses today. ``LiveEngine`` is intentionally a stub — it will be
re-introduced in a focused follow-up PR that ships the full live path
together (market entry + STOP/TP placement + manage() with OCO
sibling-cancel + reconcile-with-fills + crash-recovery integration).

The partial LiveEngine that lived here previously was reverted because
a half-built engine in production code is more dangerous than no
engine: the constructor would succeed against testnet but ``manage()``
raised ``NotImplementedError`` on the second cycle, meaning a live
trade could be opened with no exit-management path. Clean slate is
safer than mixed state.

``ReconcileReport`` and ``ReconcileMismatch`` are kept here because the
runner's ``_reconcile_or_die`` startup check uses them; they're the
seam the future LiveEngine plugs into.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

from tradingagents.solana_bot.config import BotConfig
from tradingagents.solana_bot.trade import FillEvent, OpenTrade

logger = logging.getLogger(__name__)


@dataclass
class PaperFillReport:
    """Result of a single ``manage`` call."""

    fills: List[FillEvent] = field(default_factory=list)
    realised_pnl: float = 0.0


@dataclass
class ReconcileReport:
    """Snapshot of exchange-side state for reconciliation against local state."""

    quote_balance: float
    base_balance: float
    open_orders: List[dict] = field(default_factory=list)
    is_live: bool = False  # PaperEngine returns False; future LiveEngine returns True


class ReconcileMismatch(Exception):
    """Raised at runner startup if local state and exchange state disagree."""


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

    def reconcile(self) -> ReconcileReport:
        """Paper engine has no exchange to reconcile against — returns local state."""
        return ReconcileReport(
            quote_balance=self.balance,
            base_balance=0.0,
            open_orders=[],
            is_live=False,
        )


class LiveEngine:
    """Live execution — intentionally not implemented in this release.

    Will be rebuilt as a coherent unit in a follow-up PR:

    * Constructor with withdraw + internal-transfer permission refusal
    * Tiny-capital balance ceiling check at construction *and* per trade
    * Exchange-precision rounding (lot step, tick size) before order placement
    * Min-notional check before sending to the exchange
    * Deterministic ``client_order_id`` derived from ``(trade_id, leg)`` for
      true network-retry idempotency
    * Market entry with actual-fill-price tracking (not planned-entry slippage)
    * STOP_LOSS_LIMIT (or STOP_MARKET — operator choice) placement, with the
      open-position-with-no-stop window minimised via a try/flatten-on-failure wrapper
    * TP1 + TP2 limit orders, with OCO sibling-cancel after partial fills
    * Trailing-stop replacement as the runner trail advances
    * ``reconcile()`` returning balance + open orders + position size
    * ``manage()`` polling fills via the exchange API and reporting via the same
      ``PaperFillReport`` shape the runner already consumes

    Until that PR lands, this class refuses to instantiate. The runner
    only ever constructs ``PaperEngine``.
    """

    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(
            "LiveEngine is intentionally stubbed in this release. Use PaperEngine. "
            "Live trading lands in a focused follow-up PR — see the docstring for scope."
        )
