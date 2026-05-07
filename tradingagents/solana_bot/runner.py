"""Paper-trade runner: live OHLCV from Binance, simulated fills.

Pulls the most recent closed bars at each cycle, recomputes indicators,
checks for a setup, and either opens or manages a position via the
``PaperEngine``. State is persisted on every transition.

The runner exits cleanly when the daily-loss kill switch trips. The
operator must restart the process to clear it (intentional friction).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Callable, Optional

from sol_bot import ai_filter as ai_filter_module
from tradingagents.solana_bot.config import BotConfig
from tradingagents.solana_bot.data import fetch_latest_closed_bars, sleep_until_next_candle
from tradingagents.solana_bot.execution import PaperEngine
from tradingagents.solana_bot.indicators import attach_indicators
from tradingagents.solana_bot.journal import TradeJournal
from tradingagents.solana_bot.risk import DailyLossTracker, position_size
from tradingagents.solana_bot.signals import check_long_setup
from tradingagents.solana_bot.state import BotState
from tradingagents.solana_bot.trade import OpenTrade

logger = logging.getLogger(__name__)


@dataclass
class CycleResult:
    """What happened during a single runner iteration. Useful for tests."""

    # "opened" | "managed" | "no_setup" | "ai_rejected" | "execute_disabled"
    # | "kill_switch" | "stale"
    action: str
    detail: str = ""
    fills: list = None
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
    sleeper: Callable[[str], None] = sleep_until_next_candle,
    fetcher: Optional[Callable] = None,
    ai_filter: Optional[Callable[[dict], str]] = None,
    execute_trades: Optional[bool] = None,
    journal: Optional[TradeJournal] = None,
) -> BotState:
    """Run the paper-trade loop.

    ``max_cycles`` caps the number of iterations (used by smoke tests).
    ``sleeper`` and ``fetcher`` are injectable so tests can drive the
    loop deterministically without sleeping or hitting the network.
    ``ai_filter`` and ``execute_trades`` are also injectable so tests
    can pin both gates without hitting Claude or the env. ``journal``
    is injectable so tests can point the trade ledger at tmp_path.
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
        return _run_loop(state, engine, config, fetch, ai, execute, log, sleeper, max_cycles)


def _run_loop(state, engine, config, fetch, ai, execute, log, sleeper, max_cycles):
    cycle = 0
    while True:
        if max_cycles is not None and cycle >= max_cycles:
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

        result = _process_bar(state, engine, enriched, bar, config, ai_filter=ai, execute_trades=execute)
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
            if state.open_trade.is_closed():  # type: ignore[union-attr]
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
    return CycleResult("opened", detail=f"size={size:.4f} entry={entry_price:.4f} stop={stop:.4f}")
