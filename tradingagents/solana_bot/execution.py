"""Execution engines.

``PaperEngine`` simulates fills against bar OHLC and is fine for
backtests and paper-trading burn-in. ``LiveEngine`` talks to a real
exchange (Binance spot via ccxt) and places real orders. Order
placement itself lands in a follow-up commit; this commit ships the
foundation: constructor, withdraw-permission check, tiny-capital
ceiling, and the reconcile() interface.
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, List, Optional

from tradingagents.solana_bot.config import BotConfig
from tradingagents.solana_bot.trade import FillEvent, OpenTrade

logger = logging.getLogger(__name__)


def _floor_to_step(value: float, step: float) -> float:
    """Round ``value`` DOWN to the nearest multiple of ``step``.

    Used for both price (tick size) and amount (lot size). Floor — never
    round up — so the resulting size never exceeds the intended size and
    we never accidentally bust through min notional from the wrong side.
    """
    if step <= 0:
        return value
    return math.floor(value / step) * step


class BelowMinNotional(Exception):
    """Order would be smaller than the exchange's min notional — refuse."""


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


class WithdrawPermissionEnabled(Exception):
    """Raised at LiveEngine construction if the API key has withdraw access."""


class LiveBalanceTooLarge(Exception):
    """Raised at LiveEngine construction if the account exceeds max_live_balance."""


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
        )


class LiveEngine:
    """Live execution against Binance spot via ccxt.

    Two safety gates fire at construction:

    * The API key must NOT have withdraw permission. We probe via
      ``client.fetch_account_permissions()`` (Binance exposes this);
      anything other than withdraw=False raises ``WithdrawPermissionEnabled``.
    * The account's quote-currency balance must not exceed
      ``config.max_live_balance``. This forces "tiny live capital only"
      at the start of the operator's live journey — they must raise the
      ceiling intentionally as confidence grows.

    Order placement (``open_long`` and ``manage``) lands in the next
    commit. ``reconcile()`` works today.
    """

    def __init__(
        self,
        config: BotConfig,
        api_key: str,
        api_secret: str,
        *,
        testnet: bool = True,
        client: Optional[Any] = None,
    ):
        self.config = config
        self._client = client if client is not None else self._build_client(api_key, api_secret, testnet)
        self._check_withdraw_permission()
        self._check_balance_ceiling()
        logger.info(
            "LiveEngine ready: testnet=%s symbol=%s max_live_balance=%.2f",
            testnet, config.symbol, config.max_live_balance,
        )

    @staticmethod
    def _build_client(api_key: str, api_secret: str, testnet: bool):
        import ccxt
        client = ccxt.binance({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        if testnet:
            client.set_sandbox_mode(True)
        return client

    def _check_withdraw_permission(self) -> None:
        """Refuse to operate if the API key can withdraw funds."""
        perms = self._client.fetch_account_permissions()
        # Binance returns a dict like {"withdraw": False, "trade": True, ...}.
        # Other exchanges may return different shapes; defensive lookup.
        withdraw_enabled = perms.get("withdraw", perms.get("enableWithdrawals", False))
        if withdraw_enabled:
            raise WithdrawPermissionEnabled(
                "the configured API key has withdraw permission — refusing to "
                "operate. Generate a trade-only key (no withdraw, no transfer) "
                "and restart."
            )

    def _check_balance_ceiling(self) -> None:
        """Refuse if the account holds more quote currency than the safety ceiling."""
        report = self.reconcile()
        if report.quote_balance > self.config.max_live_balance:
            raise LiveBalanceTooLarge(
                f"account quote balance ({report.quote_balance:.2f}) exceeds "
                f"max_live_balance ({self.config.max_live_balance:.2f}). "
                f"Either reduce the account balance or raise max_live_balance "
                f"in BotConfig deliberately."
            )

    def reconcile(self) -> ReconcileReport:
        balance = self._client.fetch_balance()
        base, quote = self.config.symbol.split("/")
        quote_balance = float(balance.get(quote, {}).get("free", 0.0))
        base_balance = float(balance.get(base, {}).get("free", 0.0))
        open_orders = self._client.fetch_open_orders(self.config.symbol)
        return ReconcileReport(
            quote_balance=quote_balance,
            base_balance=base_balance,
            open_orders=open_orders,
        )

    def _market(self) -> dict:
        """ccxt market metadata for the configured symbol — cached after first load."""
        markets = self._client.load_markets()
        try:
            return markets[self.config.symbol]
        except KeyError as exc:
            raise ValueError(f"unknown symbol on exchange: {self.config.symbol}") from exc

    def _round_amount(self, amount: float) -> float:
        market = self._market()
        step = float(market.get("limits", {}).get("amount", {}).get("min", 0)) or float(
            market.get("precision", {}).get("amount", 0)
        )
        return _floor_to_step(amount, step) if step else amount

    def _round_price(self, price: float) -> float:
        market = self._market()
        step = float(market.get("limits", {}).get("price", {}).get("min", 0)) or float(
            market.get("precision", {}).get("price", 0)
        )
        return _floor_to_step(price, step) if step else price

    def _check_min_notional(self, amount: float, price: float) -> None:
        market = self._market()
        min_cost = float(market.get("limits", {}).get("cost", {}).get("min", 0) or 0)
        notional = amount * price
        if min_cost and notional < min_cost:
            raise BelowMinNotional(
                f"intended notional ({notional:.4f} {self.config.symbol.split('/')[1]}) "
                f"is below the exchange minimum ({min_cost:.4f}). Either raise size or "
                f"the per-trade risk_pct."
            )

    @staticmethod
    def _client_order_id(prefix: str) -> str:
        """Generate an idempotent client order ID. Re-submitting the same ID is a no-op
        on Binance — protects against duplicate orders on retry."""
        return f"sb-{prefix}-{uuid.uuid4().hex[:16]}"

    def open_long(self, *, entry_price: float, size: float, atr_value: float) -> OpenTrade:
        """Place a market-buy entry plus a protective stop-loss order.

        TP1/TP2 placement and the OCO sibling-cancel dance land in a
        follow-up commit. The bot's runner.manage() polls for fills via
        reconcile() and handles the partial-exit + trailing logic against
        the exchange directly.
        """
        size = self._round_amount(size)
        if size <= 0:
            raise BelowMinNotional("rounded size collapsed to 0 — increase risk_pct or balance")
        rounded_entry = self._round_price(entry_price)
        self._check_min_notional(size, rounded_entry)

        stop_price = self._round_price(entry_price - self.config.atr_mult * atr_value)
        if stop_price <= 0:
            raise ValueError(f"computed stop_price={stop_price} is non-positive")

        entry_coid = self._client_order_id("entry")
        entry_order = self._client.create_order(
            symbol=self.config.symbol,
            type="market",
            side="buy",
            amount=size,
            params={"newClientOrderId": entry_coid},
        )

        # Place a protective stop-loss covering the full filled size. We use
        # STOP_LOSS_LIMIT (Binance spot) so we control the worst-case fill.
        stop_coid = self._client_order_id("stop")
        stop_order = self._client.create_order(
            symbol=self.config.symbol,
            type="STOP_LOSS_LIMIT",
            side="sell",
            amount=size,
            price=stop_price,
            params={
                "newClientOrderId": stop_coid,
                "stopPrice": stop_price,
                "timeInForce": "GTC",
            },
        )

        trade = OpenTrade.open_long(
            entry=rounded_entry,
            atr_value=atr_value,
            size=size,
            atr_mult=self.config.atr_mult,
            tp1_r=self.config.tp1_r,
            tp2_r=self.config.tp2_r,
            tp1_close_fraction=self.config.tp1_close_fraction,
            tp2_close_fraction=self.config.tp2_close_fraction,
            trail_atr_mult=self.config.trail_atr_mult,
        )

        # Surface the order IDs for the caller to persist into BotState.
        # The runner stuffs these into state.extras["live_order_ids"] so a
        # crash + restart can re-find the orders via fetch_order(coid).
        self.last_order_ids = {
            "entry": entry_coid,
            "entry_exchange_id": entry_order.get("id"),
            "stop": stop_coid,
            "stop_exchange_id": stop_order.get("id"),
        }
        logger.info(
            "LIVE LONG opened: size=%.6f entry=%.4f stop=%.4f entry_coid=%s stop_coid=%s",
            size, rounded_entry, stop_price, entry_coid, stop_coid,
        )
        return trade

    def manage(self, trade: OpenTrade, *, high: float, low: float, close: float) -> PaperFillReport:
        raise NotImplementedError(
            "LiveEngine.manage lands in the next commit (poll fills, OCO sibling-cancel "
            "after TP1, trailing stop replacement). For now use PaperEngine."
        )
