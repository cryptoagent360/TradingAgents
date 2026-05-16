"""Event-driven backtester.

Replays historical OHLCV bar-by-bar through the same signal/risk/trade
machinery the live runner uses. Entries fill at the bar close on the
signal bar; subsequent bars resolve TP and stop fills.

Outputs:
* ``trades.csv`` — one row per closed trade with entry, exit legs, PnL, R
* ``equity.csv`` — running equity (mark-to-market on each bar close)
* ``summary.json`` — aggregate stats (trades, win rate, avg R, max DD, PF)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

from tradingagents.solana_bot.config import BotConfig
from tradingagents.solana_bot.execution import PaperEngine
from tradingagents.solana_bot.indicators import attach_indicators
from tradingagents.solana_bot.risk import DailyLossTracker, position_size
from tradingagents.solana_bot.signals import check_long_setup
from tradingagents.solana_bot.trade import FillEvent, OpenTrade


@dataclass
class ClosedTrade:
    entry_ts: int
    entry_price: float
    initial_stop: float
    size: float
    fills: List[FillEvent] = field(default_factory=list)
    realised_pnl: float = 0.0

    @property
    def realised_r(self) -> float:
        risk = (self.entry_price - self.initial_stop) * self.size
        if risk <= 0:
            return 0.0
        return self.realised_pnl / risk


@dataclass
class BacktestResult:
    closed_trades: List[ClosedTrade]
    equity_curve: pd.DataFrame
    summary: dict
    output_dir: Optional[Path] = None


def _summarise(closed: List[ClosedTrade], equity: pd.DataFrame, starting_balance: float) -> dict:
    n = len(closed)
    wins = [t for t in closed if t.realised_pnl > 0]
    losses = [t for t in closed if t.realised_pnl <= 0]
    gross_profit = sum(t.realised_pnl for t in wins)
    gross_loss = -sum(t.realised_pnl for t in losses)
    if equity.empty:
        max_dd = 0.0
    else:
        running_peak = equity["equity"].cummax()
        drawdown = (equity["equity"] - running_peak) / running_peak
        max_dd = float(drawdown.min()) if not drawdown.empty else 0.0
    return {
        "trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": (len(wins) / n) if n else 0.0,
        "avg_r": (sum(t.realised_r for t in closed) / n) if n else 0.0,
        "total_pnl": sum(t.realised_pnl for t in closed),
        "ending_balance": float(equity["equity"].iloc[-1]) if not equity.empty else starting_balance,
        "max_drawdown": max_dd,
        "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0,
    }


def run_backtest(
    df: pd.DataFrame,
    config: BotConfig,
    *,
    starting_balance: float,
    output_dir: Optional[Path] = None,
) -> BacktestResult:
    """Run the backtest on a pre-fetched OHLCV DataFrame.

    ``df`` must have ``timestamp``, ``open``, ``high``, ``low``,
    ``close``, ``volume`` columns. Indicators are attached internally
    and recomputed once on the full series; signal evaluation slices
    the trailing window per bar to avoid look-ahead.
    """
    if len(df) < config.min_bars + 1:
        raise ValueError(f"need at least {config.min_bars + 1} bars, got {len(df)}")

    enriched = attach_indicators(
        df,
        ema_fast=config.ema_fast,
        ema_slow=config.ema_slow,
        rsi_period=config.rsi_period,
        atr_period=config.atr_period,
    )

    engine = PaperEngine(config=config, balance=starting_balance)
    tracker = DailyLossTracker(starting_balance=starting_balance, max_daily_loss_pct=config.max_daily_loss_pct)

    open_trade: Optional[OpenTrade] = None
    open_trade_record: Optional[ClosedTrade] = None
    closed: List[ClosedTrade] = []
    equity_rows: list[dict] = []

    for i in range(config.min_bars, len(enriched)):
        bar = enriched.iloc[i]
        ts = int(bar["timestamp"])

        # Roll over the daily-loss tracker into this bar's UTC day so realised PnL
        # accrues against the right calendar day even when wall-clock time differs.
        day = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).date().isoformat()
        if tracker.today_date_utc != day:
            tracker.today_date_utc = day
            tracker.today_pnl = 0.0
            tracker.kill_switch_triggered = False

        if open_trade is not None:
            report = engine.manage(open_trade, high=float(bar["high"]), low=float(bar["low"]), close=float(bar["close"]))
            if report.fills:
                assert open_trade_record is not None
                open_trade_record.fills.extend(report.fills)
                open_trade_record.realised_pnl += report.realised_pnl
                tracker.record_pnl(report.realised_pnl)
                if open_trade.is_closed():
                    closed.append(open_trade_record)
                    open_trade = None
                    open_trade_record = None

        if open_trade is None and tracker.can_open_trade():
            window = enriched.iloc[: i + 1]
            signal = check_long_setup(window, config)
            if signal.is_long:
                entry_price = float(bar["close"])
                atr_value = float(bar["atr"])
                stop = entry_price - config.atr_mult * atr_value
                size = position_size(engine.balance, entry_price, stop, config.risk_pct)
                open_trade = engine.open_long(entry_price=entry_price, size=size, atr_value=atr_value)
                open_trade_record = ClosedTrade(
                    entry_ts=ts,
                    entry_price=entry_price,
                    initial_stop=open_trade.initial_stop,
                    size=size,
                )

        equity_rows.append({"timestamp": ts, "equity": engine.balance})

    # Flush any still-open trade at the last bar's close so simulation length
    # doesn't dictate whether a trade counts.
    if open_trade is not None and open_trade_record is not None:
        last_close = float(enriched["close"].iloc[-1])
        flush = FillEvent(kind="eod", price=last_close, size=open_trade.remaining)
        pnl = open_trade.realised_pnl([flush], fee_rate=config.taker_fee)
        engine.balance += pnl
        open_trade.remaining = 0.0
        open_trade_record.fills.append(flush)
        open_trade_record.realised_pnl += pnl
        closed.append(open_trade_record)
        if equity_rows:
            equity_rows[-1] = {"timestamp": equity_rows[-1]["timestamp"], "equity": engine.balance}

    equity = pd.DataFrame(equity_rows)
    summary = _summarise(closed, equity, starting_balance)
    summary["starting_balance"] = starting_balance

    result = BacktestResult(closed_trades=closed, equity_curve=equity, summary=summary, output_dir=None)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_outputs(result, output_dir)
        result.output_dir = output_dir

    return result


def _write_outputs(result: BacktestResult, output_dir: Path) -> None:
    rows = []
    for t in result.closed_trades:
        for ev in t.fills:
            rows.append(
                {
                    "entry_ts": t.entry_ts,
                    "entry_price": t.entry_price,
                    "size": t.size,
                    "leg": ev.kind,
                    "exit_price": ev.price,
                    "leg_size": ev.size,
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "trades.csv", index=False)
    result.equity_curve.to_csv(output_dir / "equity.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(result.summary, indent=2))


def default_output_dir(config: BotConfig) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return config.backtests_dir / ts
