"""Execution engines.

``PaperEngine`` simulates fills against bar OHLC and is the default for
``TRADING_MODE=paper`` runs.

``LiveEngine`` talks to Kraken via ccxt and places real orders. All eight
safety items from the SRE review are wired into this implementation —
see the LiveEngine docstring for the full list. The operator is still
responsible for verifying the integration on a tiny live balance before
scaling up: real exchanges are full of edge cases (rate limits, partial
fills, order-type quirks) that mocked unit tests cannot exercise.

``ReconcileReport`` and ``ReconcileMismatch`` are shared by both engines:
``PaperEngine.reconcile`` returns local state with ``is_live=False`` so
the runner's drift check is a no-op for paper; ``LiveEngine.reconcile``
fetches real exchange state and the drift check matters.
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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


class TradeOnlyKeyNotConfirmed(Exception):
    """BotConfig.confirm_trade_only_key must be True before LiveEngine instantiates."""


class CredentialsRejected(Exception):
    """Smoke-test of API credentials (fetch_balance) failed at construction."""


class LiveBalanceTooLarge(Exception):
    """Account quote balance exceeds BotConfig.max_live_balance — refuse."""


class StopPlacementFailedFlattened(Exception):
    """Entry filled but stop placement failed; position was flattened immediately."""


class StopPlacementAndFlattenFailed(Exception):
    """Catastrophic: entry filled, stop failed to place, AND flatten failed.

    Operator must intervene manually. Raised after notify() has fired.
    """


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
    is_live: bool = False  # PaperEngine returns False; LiveEngine returns True


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
        self.balance -= entry_price * size * self.config.taker_fee
        return trade

    def manage(self, trade: OpenTrade, *, high: float, low: float, close: float) -> PaperFillReport:
        events = trade.manage(high=high, low=low, close=close)
        if not events:
            return PaperFillReport()
        pnl = trade.realised_pnl(events, fee_rate=self.config.taker_fee)
        self.balance += pnl
        return PaperFillReport(fills=events, realised_pnl=pnl)

    def reconcile(self) -> ReconcileReport:
        return ReconcileReport(
            quote_balance=self.balance,
            base_balance=0.0,
            open_orders=[],
            is_live=False,
        )


class LiveEngine:
    """Live execution against Kraken via ccxt.

    Eight safety items, all wired into this implementation:

    1. ``confirm_trade_only_key`` — refuses to instantiate unless the operator
       has set ``BotConfig.confirm_trade_only_key=True`` explicitly. Kraken
       does not expose a permissions API; this is the strongest defense short
       of refusing to operate.
    2. **Credentials smoke test** — ``fetch_balance()`` at construction; a
       bad/revoked key raises ``CredentialsRejected`` immediately rather than
       discovering the problem at first trade.
    3. **Balance ceiling at construction AND per trade** — refuses if quote
       balance exceeds ``max_live_balance`` ($100 default). Re-checked on
       every ``open_long`` so a mid-run deposit doesn't silently bust the
       invariant.
    4. **Precision rounding** — amount/price floor-rounded to the market's
       lot/tick step before any order placement. Floor (never round up) so
       size never exceeds intent.
    5. **Min notional check** — refuses below the exchange's
       ``limits.cost.min`` before any API call.
    6. **Deterministic ``clientOrderId``** derived from ``(trade_id, leg)`` —
       network-retry idempotency. A reconnect mid-placement doesn't double-place.
    7. **Stop-placement-failure → flatten** — if the protective stop fails to
       place after the entry fills, the position is immediately market-sold.
       Naked-long exposure window is closed.
    8. **Actual fill price** — entry price for R math comes from the exchange's
       reported ``average``/``price``, not the planned entry. Slippage doesn't
       silently mis-size the stop.

    Order layout after ``open_long``:

    - Market BUY (entry) — full size
    - STOP-LOSS market (stop) — full size, at entry - atr_mult * ATR
    - LIMIT SELL (tp1) — 50% of size, at entry + tp1_r * R
    - LIMIT SELL (tp2) — 25% of size, at entry + tp2_r * R

    ``manage()`` polls ``fetch_open_orders()`` each cycle; missing orders
    are detected as fills, ``FillEvent``s are emitted, and the in-memory
    ``OpenTrade`` state is updated. When TP1 fills, the original full-size
    stop is cancelled and replaced with a break-even stop for the remaining
    size. When TP2 fills, the runner is entered; subsequent bars advance
    the trailing stop and replace the resting stop order accordingly.

    .. note:: This class targets Kraken specifically. The ccxt unified API
       handles the auth + signing; order-type names (``stop-loss``,
       ``take-profit``, etc.) are Kraken-specific. The unit tests use a
       mocked ccxt client and verify the *logic*, not Kraken's actual
       order-routing behavior. Operator-side smoke testing on a $50–$100
       live balance is mandatory before scaling up.
    """

    EXCHANGE_ID = "kraken"

    def __init__(
        self,
        config: BotConfig,
        api_key: str,
        api_secret: str,
        *,
        client: Optional[Any] = None,
    ):
        if not config.confirm_trade_only_key:
            raise TradeOnlyKeyNotConfirmed(
                "BotConfig.confirm_trade_only_key must be set to True before "
                "LiveEngine will instantiate. Set it ONLY after you have "
                "verified in the exchange UI that your API key has only "
                "'Query Funds' + 'Create & Modify Orders' + 'Cancel Orders' "
                "permissions. NEVER enable 'Withdraw Funds' on this key."
            )
        self.config = config
        self._client = client if client is not None else self._build_client(api_key, api_secret)
        self._market_cache: Optional[dict] = None
        self._active_orders: Dict[str, Optional[dict]] = {}  # leg -> {coid, exchange_id, size, price}
        self._smoke_test_credentials()
        self._check_balance_ceiling()
        logger.info(
            "LiveEngine ready: exchange=%s symbol=%s max_live_balance=%.2f",
            self.EXCHANGE_ID, config.symbol, config.max_live_balance,
        )

    @staticmethod
    def _build_client(api_key: str, api_secret: str):
        import ccxt
        return ccxt.kraken({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
        })

    def _smoke_test_credentials(self) -> None:
        """Verify the API key works by fetching balance once at construction.

        A revoked, mis-typed, or read-only key fails here with a clear error
        rather than crashing on first trade.
        """
        try:
            self._client.fetch_balance()
        except Exception as exc:
            raise CredentialsRejected(
                f"fetch_balance() failed at construction: {exc!r}. "
                f"Verify the API key is correct, active, and has Query Funds + "
                f"Order permissions enabled."
            ) from exc

    def _check_balance_ceiling(self) -> None:
        report = self.reconcile()
        if report.quote_balance > self.config.max_live_balance:
            raise LiveBalanceTooLarge(
                f"account quote balance ({report.quote_balance:.2f}) exceeds "
                f"max_live_balance ({self.config.max_live_balance:.2f}). "
                f"Reduce the account balance or raise max_live_balance in "
                f"BotConfig deliberately. The ceiling is the 'tiny live capital "
                f"only' invariant — do not raise it lightly."
            )

    def reconcile(self) -> ReconcileReport:
        balance = self._client.fetch_balance()
        base, quote = self.config.symbol.split("/")
        quote_balance = float(balance.get(quote, {}).get("free", 0.0) or 0.0)
        base_balance = float(balance.get(base, {}).get("free", 0.0) or 0.0)
        open_orders = self._client.fetch_open_orders(self.config.symbol)
        return ReconcileReport(
            quote_balance=quote_balance,
            base_balance=base_balance,
            open_orders=open_orders,
            is_live=True,
        )

    def _market(self) -> dict:
        if self._market_cache is not None:
            return self._market_cache
        markets = self._client.load_markets()
        if self.config.symbol not in markets:
            raise ValueError(
                f"unknown symbol on {self.EXCHANGE_ID}: {self.config.symbol}. "
                f"Check BOT_SYMBOL — Kraken uses e.g. SOL/USD or SOL/EUR, not SOL/USDT."
            )
        self._market_cache = markets[self.config.symbol]
        return self._market_cache

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
                f"is below the exchange minimum ({min_cost:.4f}). Either raise "
                f"size or the per-trade risk_pct."
            )

    @staticmethod
    def _coid(trade_id: str, leg: str) -> str:
        """Deterministic client_order_id — retries with the same (trade_id, leg)
        are idempotent on Kraken."""
        return f"sb-{trade_id}-{leg}"

    def open_long(
        self,
        *,
        entry_price: float,
        size: float,
        atr_value: float,
        trade_id: Optional[str] = None,
    ) -> OpenTrade:
        # Per-trade balance recheck. Constructor was a one-shot; this guards
        # against deposits / accumulated PnL silently busting the ceiling.
        self._check_balance_ceiling()

        size = self._round_amount(size)
        if size <= 0:
            raise BelowMinNotional(
                "rounded size collapsed to 0 — increase risk_pct or account balance"
            )
        rounded_entry = self._round_price(entry_price)
        self._check_min_notional(size, rounded_entry)

        if trade_id is None:
            trade_id = uuid.uuid4().hex[:12]
        entry_coid = self._coid(trade_id, "entry")
        stop_coid = self._coid(trade_id, "stop")
        tp1_coid = self._coid(trade_id, "tp1")
        tp2_coid = self._coid(trade_id, "tp2")

        # 1. Market entry. Use actual fill price for downstream R math —
        # market orders slip and using rounded_entry would mis-size everything.
        entry_order = self._client.create_order(
            symbol=self.config.symbol,
            type="market",
            side="buy",
            amount=size,
            params={"clientOrderId": entry_coid},
        )
        actual_entry = float(
            entry_order.get("average") or entry_order.get("price") or rounded_entry
        )

        stop_price = self._round_price(actual_entry - self.config.atr_mult * atr_value)
        if stop_price <= 0:
            self._panic_flatten(size, trade_id, reason="computed stop_price <= 0")
            raise ValueError(f"stop_price={stop_price} non-positive after entry")

        # 2. Protective stop. CRITICAL: failure here means the position is
        # naked. Flatten immediately rather than holding an unprotected long.
        try:
            stop_order = self._client.create_order(
                symbol=self.config.symbol,
                type="stop-loss",  # Kraken: market stop, guaranteed fill
                side="sell",
                amount=size,
                params={"clientOrderId": stop_coid, "stopPrice": stop_price},
            )
        except Exception as stop_exc:
            logger.error("stop placement failed: %s — flattening position", stop_exc)
            self._panic_flatten(size, trade_id, reason=f"stop placement failed: {stop_exc}")
            raise StopPlacementFailedFlattened(
                f"stop placement failed ({stop_exc}); position flattened"
            ) from stop_exc

        # 3 + 4. TP1 and TP2 limits. Less critical than the stop — if these
        # fail, the position is still safe (stop is in place) and we lose
        # only the convenience of pre-positioned exits. Log and continue.
        r_per_unit = actual_entry - stop_price
        tp1_price = self._round_price(actual_entry + self.config.tp1_r * r_per_unit)
        tp2_price = self._round_price(actual_entry + self.config.tp2_r * r_per_unit)
        tp1_size = self._round_amount(size * self.config.tp1_close_fraction)
        tp2_size = self._round_amount(size * self.config.tp2_close_fraction)

        tp1_order = self._try_place_limit("tp1", tp1_coid, tp1_size, tp1_price)
        tp2_order = self._try_place_limit("tp2", tp2_coid, tp2_size, tp2_price)

        # Build the in-memory OpenTrade using actual_entry so stop/TP/trail math
        # is computed against what we actually paid.
        trade = OpenTrade.open_long(
            entry=actual_entry,
            atr_value=atr_value,
            size=size,
            atr_mult=self.config.atr_mult,
            tp1_r=self.config.tp1_r,
            tp2_r=self.config.tp2_r,
            tp1_close_fraction=self.config.tp1_close_fraction,
            tp2_close_fraction=self.config.tp2_close_fraction,
            trail_atr_mult=self.config.trail_atr_mult,
        )
        # Override prices to match what we actually placed on exchange (after
        # rounding to tick) so manage() compares correctly.
        trade.initial_stop = stop_price
        trade.current_stop = stop_price
        trade.tp1_price = tp1_price
        trade.tp2_price = tp2_price

        self._active_orders = {
            "stop": {"coid": stop_coid, "exchange_id": stop_order.get("id"),
                     "size": size, "price": stop_price},
            "tp1": ({"coid": tp1_coid, "exchange_id": tp1_order.get("id"),
                     "size": tp1_size, "price": tp1_price} if tp1_order else None),
            "tp2": ({"coid": tp2_coid, "exchange_id": tp2_order.get("id"),
                     "size": tp2_size, "price": tp2_price} if tp2_order else None),
        }
        self.last_order_ids = {
            "trade_id": trade_id,
            "entry": entry_coid,
            "entry_exchange_id": entry_order.get("id"),
            "stop": stop_coid,
            "stop_exchange_id": stop_order.get("id"),
            "tp1": tp1_coid,
            "tp1_exchange_id": tp1_order.get("id") if tp1_order else None,
            "tp2": tp2_coid,
            "tp2_exchange_id": tp2_order.get("id") if tp2_order else None,
        }
        logger.info(
            "LIVE LONG opened: trade_id=%s size=%.6f entry=%.6f stop=%.6f "
            "tp1=%.6f tp2=%.6f (planned_entry=%.6f slippage=%.6f)",
            trade_id, size, actual_entry, stop_price, tp1_price, tp2_price,
            rounded_entry, actual_entry - rounded_entry,
        )
        return trade

    def _try_place_limit(self, leg: str, coid: str, size: float, price: float) -> Optional[dict]:
        """Place a TP limit order. Logs and returns None on failure (best-effort)."""
        try:
            return self._client.create_order(
                symbol=self.config.symbol,
                type="limit",
                side="sell",
                amount=size,
                price=price,
                params={"clientOrderId": coid},
            )
        except Exception as exc:
            logger.warning("%s limit placement failed: %s — continuing without it", leg, exc)
            return None

    def _panic_flatten(self, size: float, trade_id: str, *, reason: str) -> None:
        """Market-sell the full position. Used when stop placement failed.

        Raises ``StopPlacementAndFlattenFailed`` if the flatten itself fails,
        which is a true emergency — operator must intervene manually.
        """
        flatten_coid = self._coid(trade_id, "flatten")
        try:
            self._client.create_order(
                symbol=self.config.symbol,
                type="market",
                side="sell",
                amount=size,
                params={"clientOrderId": flatten_coid},
            )
            logger.warning("position flattened (reason: %s)", reason)
        except Exception as flat_exc:
            logger.critical(
                "FLATTEN FAILED after %s — manual intervention required: %s",
                reason, flat_exc,
            )
            raise StopPlacementAndFlattenFailed(
                f"could not flatten after stop failure ({reason}): {flat_exc}"
            ) from flat_exc

    def manage(self, trade: OpenTrade, *, high: float, low: float, close: float) -> PaperFillReport:
        """Poll exchange for fills; update OpenTrade; emit FillEvents.

        Strategy: fetch open orders; any order in self._active_orders whose
        clientOrderId is no longer in the open-orders list is assumed filled.
        Update trade state accordingly (TP1 → BE stop; TP2 → enter trail mode).
        For the trailing leg, advance the resting stop when the trail advances.
        """
        events: List[FillEvent] = []
        try:
            open_orders = self._client.fetch_open_orders(self.config.symbol)
        except Exception as exc:
            # A failed poll should not crash the runner; the next cycle will retry.
            logger.warning("fetch_open_orders failed: %s — skipping manage()", exc)
            return PaperFillReport()
        open_coids = {o.get("clientOrderId") for o in open_orders if o.get("clientOrderId")}

        # Detect filled legs by their absence from the open list.
        for leg in ("stop", "tp1", "tp2"):
            info = self._active_orders.get(leg)
            if info is None or info["coid"] in open_coids:
                continue
            events.append(FillEvent(kind=leg, price=info["price"], size=info["size"]))
            self._active_orders[leg] = None
            trade.remaining = max(0.0, trade.remaining - info["size"])
            if leg == "tp1":
                trade.tp1_hit = True
                # Cancel the old full-size stop and place a BE stop for the remaining size.
                self._replace_stop(trade, new_stop_price=self._round_price(trade.entry))
            elif leg == "tp2":
                trade.tp2_hit = True
            elif leg == "stop":
                trade.remaining = 0.0

        # Trailing: once we're past TP2 and the trail advances, cancel + replace
        # the resting stop. Only do this if the high actually moved.
        if trade.tp2_hit and trade.remaining > 1e-9:
            trade.trail_high = max(trade.trail_high, high)
            new_trail = self._round_price(
                trade.trail_high - trade.trail_atr_mult * trade.atr_at_entry
            )
            if new_trail > trade.current_stop:
                trade.current_stop = new_trail
                self._replace_stop(trade, new_stop_price=new_trail)

        if not events:
            return PaperFillReport()
        pnl = trade.realised_pnl(events, fee_rate=self.config.taker_fee)
        return PaperFillReport(fills=events, realised_pnl=pnl)

    def _replace_stop(self, trade: OpenTrade, *, new_stop_price: float) -> None:
        """Cancel the current resting stop and place a new one for ``trade.remaining``.

        Best-effort: a failure here logs and leaves whatever stop exists in
        place; the runner does not crash. If both old-cancel and new-place
        fail, the operator will see the alert in logs.
        """
        old_info = self._active_orders.get("stop")
        if old_info is not None:
            try:
                self._client.cancel_order(old_info["exchange_id"], self.config.symbol)
            except Exception as exc:
                logger.warning("stop cancel failed: %s — leaving in place", exc)
        if trade.remaining <= 0:
            self._active_orders["stop"] = None
            return
        new_coid = self._coid(self.last_order_ids["trade_id"], f"stop-{int(new_stop_price*1000)}")
        try:
            order = self._client.create_order(
                symbol=self.config.symbol,
                type="stop-loss",
                side="sell",
                amount=trade.remaining,
                params={"clientOrderId": new_coid, "stopPrice": new_stop_price},
            )
            self._active_orders["stop"] = {
                "coid": new_coid,
                "exchange_id": order.get("id"),
                "size": trade.remaining,
                "price": new_stop_price,
            }
        except Exception as exc:
            logger.error(
                "replacement stop placement failed at %.6f: %s — position may be unprotected",
                new_stop_price, exc,
            )
            self._active_orders["stop"] = None
