"""Paper-trade runner: live OHLCV from Binance, simulated fills.

Pulls the most recent closed bars at each cycle, recomputes indicators,
checks for a setup, and either opens or manages a position via the
``PaperEngine``. State is persisted on every transition.

The runner exits cleanly when the daily-loss kill switch trips. The
operator must restart the process to clear it (intentional friction).
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, List, Literal, Optional

from sol_bot import ai_filter as ai_filter_module
from tradingagents.solana_bot.config import BotConfig
from tradingagents.solana_bot.data import fetch_latest_closed_bars, sleep_until_next_candle
from tradingagents.solana_bot.execution import PaperEngine, ReconcileMismatch
from tradingagents.solana_bot.indicators import attach_indicators
from tradingagents.solana_bot.journal import TradeJournal
from tradingagents.solana_bot.risk import DailyLossTracker, position_size
from tradingagents.solana_bot.signals import check_long_setup
from tradingagents.solana_bot.state import BotState
from tradingagents.solana_bot.trade import OpenTrade

logger = logging.getLogger(__name__)


CycleAction = Literal[
    "opened", "managed", "no_setup", "ai_rejected", "execute_disabled",
    "kill_switch", "stale",
]


@dataclass
class CycleResult:
    """What happened during a single runner iteration. Useful for tests."""

    action: CycleAction
    detail: str = ""
    # Default factory, not a literal None — `for fill in result.fills:` must be
    # safe on every CycleResult, not only the ones with actual fills.
    fills: List = field(default_factory=list)
    pnl: float = 0.0


def _build_ai_market_data(bar, _config: BotConfig) -> dict:
    close = float(bar["close"])
    ema_fast = float(bar["ema_fast"])
    distance_pct = (close - ema_fast) / ema_fast * 100.0
    return {
        "trend": "bullish (price > EMA200, EMA50 > EMA200)",
        "rsi": f"{float(bar['rsi']):.1f}",
        "distance": f"{distance_pct:+.2f}%",
        "volume": "spike confirmed",
    }


def run_paper(
    config: BotConfig,
    *,
    starting_balance: float,
    max_cycles: Optional[int] = None,
    max_runtime_s: Optional[float] = None,
    sleeper: Callable[[str], None] = sleep_until_next_candle,
    fetcher: Optional[Callable] = None,
    ai_filter: Optional[Callable[[dict], str]] = None,
    execute_trades: Optional[bool] = None,
    journal: Optional[TradeJournal] = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> BotState:
    """Run the paper-trade loop.

    ``max_cycles`` caps the number of iterations (used by smoke tests).
    ``max_runtime_s`` is the wall-clock budget — useful for burn-in runs
    (e.g. 72h before promoting to live). Whichever limit fires first
    ends the loop. ``sleeper`` and ``fetcher`` are injectable so tests
    can drive the loop deterministically without sleeping or hitting
    the network. ``ai_filter`` and ``execute_trades`` are also
    injectable so tests can pin both gates without hitting Claude or
    the env. ``journal`` is injectable so tests can point the trade
    ledger at tmp_path. ``monotonic`` is injectable so tests can
    fast-forward the burn-in clock.
    """
    state = BotState.load(config.state_path)
    if state.tracker is None:
        state.tracker = DailyLossTracker(starting_balance=starting_balance, max_daily_loss_pct=config.max_daily_loss_pct)
    engine = PaperEngine(config=config, balance=starting_balance + state.tracker.today_pnl)
    fetch = fetcher if fetcher is not None else (lambda cfg, n: fetch_latest_closed_bars(cfg, n))
    ai = ai_filter if ai_filter is not None else ai_filter_module.ai_trade_filter
    execute = execute_trades if execute_trades is not None else ai_filter_module.EXECUTE_TRADES
    log = journal if journal is not None else TradeJournal(config.journal_path)

    with state.acquire_runner_lock():
        _reconcile_or_die(state, engine)
        return _run_loop(
            state, engine, config, fetch, ai, execute, log, sleeper,
            max_cycles, max_runtime_s, monotonic,
        )


def _reconcile_or_die(state: BotState, engine) -> None:
    """Reconcile local state against the exchange before the loop starts.

    Three drift cases against a live engine:
      A. State has a trade and exchange has a position → matched, continue.
      B. State has a trade but exchange has no position → trade closed
         while we were down. Clear the local trade and continue.
      C. State has no trade but exchange has a position → DRIFT. Refuse
         to operate (ReconcileMismatch) — could be a position the
         operator opened manually, or a bug. Fail closed.

    PaperEngine reports is_live=False; drift detection is skipped.
    """
    report = engine.reconcile()
    if not report.is_live:
        return
    state_has_trade = state.has_open_trade()
    exchange_has_position = report.base_balance > 0 or len(report.open_orders) > 0
    if state_has_trade and not exchange_has_position:
        logger.warning(
            "RECONCILE: state shows open trade but exchange has none — assuming "
            "trade closed during downtime; clearing local trade"
        )
        state.open_trade = None
        state.extras.pop("current_trade_id", None)
        state.extras.pop("current_trade_pnl", None)
        state.save()
    elif not state_has_trade and exchange_has_position:
        raise ReconcileMismatch(
            f"exchange shows base balance {report.base_balance:.6f} or "
            f"{len(report.open_orders)} open orders, but local state has no "
            f"open trade. Refusing to start; investigate manually before clearing."
        )
    else:
        logger.info("reconcile OK: state and exchange agree")


def _run_loop(state, engine, config, fetch, ai, execute, journal, sleeper, max_cycles, max_runtime_s, monotonic):
    cycle = 0
    started = monotonic()
    while True:
        if max_cycles is not None and cycle >= max_cycles:
            return state
        if max_runtime_s is not None and monotonic() - started >= max_runtime_s:
            logger.info("burn-in complete: ran %.1fs (limit=%.1fs)", monotonic() - started, max_runtime_s)
            state.extras["last_cycle"] = "burn_in_complete"
            state.save()
            return state
        cycle += 1

        if not state.tracker.can_open_trade() and not state.has_open_trade():
            logger.warning("kill switch active and no open trade — exiting loop")
            state.extras["last_cycle"] = "kill_switch_exit"
            state.save()
            return state

        df = fetch(config, config.min_bars + 5)
        if df.empty:
            logger.warning("no bars returned; sleeping")
            sleeper(config.timeframe)
            continue
        latest_ts = int(df["timestamp"].iloc[-1])
        if state.last_candle_ts is not None and latest_ts <= state.last_candle_ts:
            sleeper(config.timeframe)
            continue
        state.last_candle_ts = latest_ts

        enriched = attach_indicators(
            df,
            ema_fast=config.ema_fast,
            ema_slow=config.ema_slow,
            rsi_period=config.rsi_period,
            atr_period=config.atr_period,
        )
        bar = enriched.iloc[-1]

        result = _process_bar(state, engine, enriched, bar, config, ai_filter=ai, execute_trades=execute, journal=journal)
        state.save()
        logger.info("cycle %d: %s — %s", cycle, result.action, result.detail)

        if not state.tracker.can_open_trade() and not state.has_open_trade():
            return state

        if max_cycles is None:
            sleeper(config.timeframe)


def _process_bar(
    state: BotState,
    engine: PaperEngine,
    enriched,
    bar,
    config: BotConfig,
    *,
    ai_filter: Callable[[dict], str],
    execute_trades: bool,
    journal: TradeJournal,
) -> CycleResult:
    if state.has_open_trade():
        report = engine.manage(
            state.open_trade,  # type: ignore[arg-type]
            high=float(bar["high"]),
            low=float(bar["low"]),
            close=float(bar["close"]),
        )
        if report.fills:
            state.tracker.record_pnl(report.realised_pnl)  # type: ignore[union-attr]
            trade_id = state.extras.get("current_trade_id", "")
            for ev in report.fills:
                journal.record_fill(
                    trade_id=trade_id,
                    symbol=config.symbol,
                    fill=ev,
                    realised_pnl=report.realised_pnl / max(len(report.fills), 1),
                )
            state.extras["current_trade_pnl"] = (
                state.extras.get("current_trade_pnl", 0.0) + report.realised_pnl
            )
            if state.open_trade.is_closed():  # type: ignore[union-attr]
                # Close the trade lifecycle: write the close event then drop
                # the trade-id keys from extras so the next open gets fresh ones.
                total_pnl = state.extras.get("current_trade_pnl", report.realised_pnl)
                trade = state.open_trade
                r_per_unit = trade.entry - trade.initial_stop  # type: ignore[union-attr]
                r_multiple = total_pnl / (r_per_unit * trade.size) if r_per_unit > 0 and trade.size > 0 else None  # type: ignore[union-attr]
                journal.record_close(
                    trade_id=trade_id,
                    symbol=config.symbol,
                    total_pnl=total_pnl,
                    r_multiple=r_multiple,
                )
                state.extras.pop("current_trade_id", None)
                state.extras.pop("current_trade_pnl", None)
                state.open_trade = None
            return CycleResult("managed", detail=f"{len(report.fills)} fill(s)", fills=report.fills, pnl=report.realised_pnl)
        return CycleResult("managed", detail="no fills this bar")

    if not state.tracker.can_open_trade():  # type: ignore[union-attr]
        return CycleResult("kill_switch", detail="daily loss limit reached")

    signal = check_long_setup(enriched, config)
    if not signal.is_long:
        return CycleResult("no_setup", detail=signal.reason)

    market_data = _build_ai_market_data(bar, config)
    decision = ai_filter(market_data)
    logger.info("AI filter decision: %s", decision)
    if decision != "APPROVE":
        return CycleResult("ai_rejected", detail=f"AI {decision}")

    if not execute_trades:
        logger.info("SIMULATION: would open LONG (EXECUTE_TRADES disabled)")
        return CycleResult("execute_disabled", detail="EXECUTE_TRADES disabled")

    entry_price = float(bar["close"])
    atr_value = float(bar["atr"])
    stop = entry_price - config.atr_mult * atr_value
    size = position_size(engine.balance, entry_price, stop, config.risk_pct)
    trade: OpenTrade = engine.open_long(entry_price=entry_price, size=size, atr_value=atr_value)
    state.open_trade = trade
    trade_id = uuid.uuid4().hex
    state.extras["current_trade_id"] = trade_id
    state.extras["current_trade_pnl"] = 0.0
    # LiveEngine surfaces .last_order_ids (entry / stop / etc. client_order_ids
    # plus exchange IDs); persist them so a crash + restart can re-find the
    # orders via fetch_order(). PaperEngine has no such attr — getattr returns
    # None and the key is omitted from extras.
    live_order_ids = getattr(engine, "last_order_ids", None)
    if live_order_ids:
        state.extras["live_order_ids"] = dict(live_order_ids)
    journal.record_open(
        trade_id=trade_id,
        symbol=config.symbol,
        trade=trade,
    )
    return CycleResult("opened", detail=f"size={size:.4f} entry={entry_price:.4f} stop={stop:.4f}")
