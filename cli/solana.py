"""Typer subcommand group for the Solana trend+pullback bot."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from tradingagents.solana_bot.backtest import default_output_dir, run_backtest
from tradingagents.solana_bot.config import BotConfig
from tradingagents.solana_bot.data import fetch_ohlcv_range
from tradingagents.solana_bot.runner import run_paper

solana_app = typer.Typer(name="solana", help="Solana trend+pullback bot (backtest, paper trading)")
console = Console()


def _make_config(symbol: str, timeframe: str, exchange: str) -> BotConfig:
    return BotConfig(symbol=symbol, timeframe=timeframe, exchange=exchange)


def _print_summary(summary: dict) -> None:
    table = Table(title="Backtest summary", show_header=True, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Trades", str(summary["trades"]))
    table.add_row("Win rate", f"{summary['win_rate']:.1%}")
    table.add_row("Avg R", f"{summary['avg_r']:.3f}")
    table.add_row("Profit factor", f"{summary['profit_factor']:.2f}" if summary["profit_factor"] != float("inf") else "∞")
    table.add_row("Total PnL", f"{summary['total_pnl']:.2f}")
    table.add_row("Ending balance", f"{summary['ending_balance']:.2f}")
    table.add_row("Max drawdown", f"{summary['max_drawdown']:.2%}")
    console.print(table)


@solana_app.command("backtest")
def backtest(
    symbol: str = typer.Option("SOL/USDT", "--symbol", help="Trading pair, e.g. SOL/USDT"),
    timeframe: str = typer.Option("1h", "--timeframe", help="Candle timeframe (1m, 5m, 15m, 1h, 4h, 1d)"),
    start: str = typer.Option(..., "--start", help="Start date YYYY-MM-DD (UTC)"),
    end: Optional[str] = typer.Option(None, "--end", help="End date YYYY-MM-DD (UTC); defaults to now"),
    balance: float = typer.Option(10000.0, "--balance", help="Starting account balance"),
    exchange: str = typer.Option("binance", "--exchange", help="ccxt exchange id"),
    output_dir: Optional[Path] = typer.Option(None, "--output", help="Override output directory"),
) -> None:
    """Replay historical OHLCV through the strategy and write trade/equity CSVs."""
    config = _make_config(symbol=symbol, timeframe=timeframe, exchange=exchange)
    console.print(f"[cyan]Fetching {symbol} {timeframe} from {start} to {end or 'now'}...[/cyan]")
    try:
        df = fetch_ohlcv_range(config, start=start, end=end)
    except Exception as exc:
        console.print(f"[red]Failed to fetch OHLCV: {exc}[/red]")
        console.print(f"[dim]Cached data (if any) lives at {config.cache_dir}.[/dim]")
        raise typer.Exit(code=1)
    if df.empty:
        console.print(f"[red]No data returned for {symbol} {timeframe} in range.[/red]")
        raise typer.Exit(code=1)
    console.print(f"[green]Loaded {len(df)} bars.[/green]")

    out_dir = output_dir or default_output_dir(config)
    result = run_backtest(df, config, starting_balance=balance, output_dir=out_dir)
    _print_summary(result.summary)
    console.print(f"\n[dim]Outputs written to:[/dim] {out_dir}")


@solana_app.command("paper")
def paper(
    symbol: str = typer.Option("SOL/USDT", "--symbol"),
    timeframe: str = typer.Option("1h", "--timeframe"),
    balance: float = typer.Option(10000.0, "--balance"),
    exchange: str = typer.Option("binance", "--exchange"),
    cycles: Optional[int] = typer.Option(None, "--cycles", help="Run only N cycles then exit (for smoke testing)"),
    burn_in_hours: Optional[float] = typer.Option(
        None, "--burn-in-hours", help="Run for at most N hours then exit cleanly (for pre-live burn-in)"
    ),
) -> None:
    """Run the paper-trade loop against live OHLCV with simulated fills."""
    config = _make_config(symbol=symbol, timeframe=timeframe, exchange=exchange)
    console.print(f"[cyan]Starting paper run for {symbol} {timeframe} (balance={balance}).[/cyan]")
    if burn_in_hours is not None:
        console.print(f"[cyan]Burn-in budget: {burn_in_hours:.1f} hours.[/cyan]")
    console.print(f"[dim]State file: {config.state_path}[/dim]\n")
    # Paper fills are simulated, so the live-execution kill switch does
    # not apply here — flip it on at the runner level for paper mode.
    state = run_paper(
        config,
        starting_balance=balance,
        max_cycles=cycles,
        max_runtime_s=burn_in_hours * 3600 if burn_in_hours is not None else None,
        execute_trades=True,
    )
    if state.tracker is not None:
        console.print(
            f"\n[bold]Today PnL:[/bold] {state.tracker.today_pnl:.2f} "
            f"(kill switch: {state.tracker.kill_switch_triggered})"
        )
    if state.has_open_trade():
        ot = state.open_trade
        console.print(
            f"[bold]Open trade:[/bold] entry={ot.entry:.4f} stop={ot.current_stop:.4f} "
            f"tp1={ot.tp1_price:.4f} tp2={ot.tp2_price:.4f} remaining={ot.remaining:.4f}"
        )
    else:
        console.print("[bold]No open trade.[/bold]")


@solana_app.command("live")
def live(
    symbol: str = typer.Option("SOL/USD", "--symbol", help="Kraken pair, e.g. SOL/USD"),
    timeframe: str = typer.Option("1h", "--timeframe"),
    balance: float = typer.Option(100.0, "--balance", help="Starting balance for bookkeeping (does NOT override exchange balance)"),
    cycles: Optional[int] = typer.Option(None, "--cycles", help="Run only N cycles then exit (smoke test)"),
    confirm: bool = typer.Option(
        False, "--i-have-verified-trade-only-key",
        help="REQUIRED. Acknowledges you have manually verified the Kraken API key has Trade permission only — no Withdraw, no Account Management.",
    ),
) -> None:
    """Run the live-trade loop against Kraken with REAL money.

    Refuses to start unless:
    * EXECUTE_TRADES = True is committed in sol_bot/ai_filter.py
    * --i-have-verified-trade-only-key flag is passed
    * KRAKEN_API_KEY / KRAKEN_API_SECRET present in environment
    * Account quote balance is at or below BotConfig.max_live_balance
    """
    from sol_bot.ai_filter import EXECUTE_TRADES
    from tradingagents.solana_bot.execution import LiveEngine
    from tradingagents.solana_bot.runner import run_paper as run_loop

    if not EXECUTE_TRADES:
        console.print("[red]EXECUTE_TRADES is False in sol_bot/ai_filter.py. Edit + commit before going live.[/red]")
        raise typer.Exit(code=2)
    if not confirm:
        console.print("[red]Refusing to start without --i-have-verified-trade-only-key.[/red]")
        raise typer.Exit(code=2)
    api_key = os.environ.get("KRAKEN_API_KEY", "").strip()
    api_secret = os.environ.get("KRAKEN_API_SECRET", "").strip()
    if not api_key or not api_secret:
        console.print("[red]KRAKEN_API_KEY and KRAKEN_API_SECRET must be set in environment.[/red]")
        raise typer.Exit(code=2)

    config = BotConfig(symbol=symbol, timeframe=timeframe, exchange="kraken", confirm_trade_only_key=True)
    console.print(f"[red bold]LIVE MODE[/red bold] {symbol} {timeframe} — max_live_balance={config.max_live_balance:.2f}")
    engine = LiveEngine(config=config, api_key=api_key, api_secret=api_secret)
    state = run_loop(config, starting_balance=balance, max_cycles=cycles, engine=engine, execute_trades=True)
    if state.tracker is not None:
        console.print(
            f"\n[bold]Today PnL:[/bold] {state.tracker.today_pnl:.2f} "
            f"(kill switch: {state.tracker.kill_switch_triggered})"
        )


__all__ = ["solana_app"]
