"""Paper-trade runner: live OHLCV from Binance, simulated fills.

Pulls the most recent closed bars at each cycle, recomputes indicators,
checks for a setup, and either opens or manages a position via the
``PaperEngine``. State is persisted on every transition.

The runner exits cleanly when the daily-loss kill switch trips. The
operator must restart the process to clear it (intentional friction).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

from tradingagents.solana_bot.config import BotConfig
from tradingagents.solana_bot.data import fetch_latest_closed_bars, sleep_until_next_candle
from tradingagents.solana_bot.execution import PaperEngine
from tradingagents.solana_bot.indicators import attach_indicators
from tradingagents.solana_bot.risk import DailyLossTracker, position_size
from tradingagents.solana_bot.signals import check_long_setup
from tradingagents.solana_bot.state import BotState
from tradingagents.solana_bot.trade import OpenTrade

logger = logging.getLogger(__name__)


@dataclass
class CycleResult:
    """What happened during a single runner iteration. Useful for tests."""

    action: str  # "opened" | "managed" | "no_setup" | "kill_switch" | "stale"
    detail: str = ""
    fills: list = None
    pnl: float = 0.0


def run_paper(
    config: BotConfig,
    *,
    starting_balance: float,
    max_cycles: Optional[int] = None,
    sleeper: Callable[[str], None] = sleep_until_next_candle,
    fetcher: Optional[Callable] = None,
) -> BotState:
    """Run the paper-trade loop.

    ``max_cycles`` caps the number of iterations (used by smoke tests).
    ``sleeper`` and ``fetcher`` are injectable so tests can drive the
    loop deterministically without sleeping or hitting the network.
    """
    state = BotState.load(config.state_path)
    if state.tracker is None:
        state.tracker = DailyLossTracker(starting_balance=starting_balance, max_daily_loss_pct=config.max_daily_loss_pct)
    engine = PaperEngine(config=config, balance=starting_balance + state.tracker.today_pnl)
    fetch = fetcher if fetcher is not None else (lambda cfg, n: fetch_latest_closed_bars(cfg, n))

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

        result = _process_bar(state, engine, enriched, bar, config)
        state.save()
        logger.info("cycle %d: %s — %s", cycle, result.action, result.detail)

        if not state.tracker.can_open_trade() and not state.has_open_trade():
            return state

        if max_cycles is None:
            sleeper(config.timeframe)


def _process_bar(state: BotState, engine: PaperEngine, enriched, bar, config: BotConfig) -> CycleResult:
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

    entry_price = float(bar["close"])
    atr_value = float(bar["atr"])
    stop = entry_price - config.atr_mult * atr_value
    size = position_size(engine.balance, entry_price, stop, config.risk_pct)
    trade: OpenTrade = engine.open_long(entry_price=entry_price, size=size, atr_value=atr_value)
    state.open_trade = trade
    return CycleResult("opened", detail=f"size={size:.4f} entry={entry_price:.4f} stop={stop:.4f}")
