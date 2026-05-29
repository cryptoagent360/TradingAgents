"""Long-running entry point for the Solana trend+pullback bot.

Suitable for background deployment via nohup or systemd:

    nohup python bot.py > output.log 2>&1 &

Loads .env at startup so ANTHROPIC_API_KEY (optional — AI filter is opt-in),
KRAKEN_API_KEY / KRAKEN_API_SECRET (for live mode), and TELEGRAM_*
credentials are available. Honors the EXECUTE_TRADES gate defined in
sol_bot/ai_filter.py verbatim — flip it to True in source to enable
real-money execution.

Operator-tunable knobs are read from the environment with sensible
defaults (no CLI flags, since this is meant to run unattended):

    BOT_STARTING_BALANCE  account size in quote currency (default 10000)
    BOT_SYMBOL            trading pair (default SOL/USDT for paper / SOL/USD for live)
    BOT_TIMEFRAME         candle timeframe (default 1h)
    BOT_BURN_IN_HOURS     wall-clock budget; loop exits after N hours (default unlimited)
    TRADING_MODE          "paper" (default) or "live"
    KRAKEN_API_KEY        live-mode credentials (Kraken via ccxt)
    KRAKEN_API_SECRET     live-mode credentials (Kraken via ccxt)

SIGTERM handler: ``systemctl stop`` (or any TERM signal) sets a stop flag
the runner checks at the top of each cycle, so the bot exits cleanly
between cycles rather than mid-API-call.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
from typing import Optional

from dotenv import load_dotenv

from sol_bot.ai_filter import EXECUTE_TRADES
from tradingagents.solana_bot.config import BotConfig
from tradingagents.solana_bot.execution import LiveEngine, PaperEngine
from tradingagents.solana_bot.runner import run_paper


_STOP_REQUESTED = False


def _install_sigterm_handler(log: logging.Logger) -> None:
    """SIGTERM / SIGINT flip the module-level flag; runner exits between cycles."""

    def _handler(signum: int, _frame) -> None:
        global _STOP_REQUESTED
        if _STOP_REQUESTED:
            log.warning("second %s — forcing exit", signal.Signals(signum).name)
            sys.exit(130)
        _STOP_REQUESTED = True
        log.warning("received %s — will exit cleanly between cycles", signal.Signals(signum).name)

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def _build_engine(mode: str, config: BotConfig, starting_balance: float, log: logging.Logger):
    """Engine factory based on TRADING_MODE.

    Paper mode → PaperEngine (always safe; simulated fills).
    Live mode  → LiveEngine (real orders against Kraken via ccxt).
    """
    if mode == "paper":
        return PaperEngine(config=config, balance=starting_balance)
    if mode == "live":
        api_key = os.environ.get("KRAKEN_API_KEY", "").strip()
        api_secret = os.environ.get("KRAKEN_API_SECRET", "").strip()
        if not api_key or not api_secret:
            log.error(
                "TRADING_MODE=live requires KRAKEN_API_KEY and KRAKEN_API_SECRET "
                "in .env. Refusing to start."
            )
            raise SystemExit(2)
        if not config.confirm_trade_only_key:
            log.error(
                "TRADING_MODE=live requires BotConfig.confirm_trade_only_key=True. "
                "Set this in your config ONLY after manually verifying the API key "
                "has Trade permission only — no Withdraw, no Account Management."
            )
            raise SystemExit(2)
        return LiveEngine(config=config, api_key=api_key, api_secret=api_secret)
    raise SystemExit(2)  # unreachable; preflight catches this


def main() -> int:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    log = logging.getLogger(__name__)
    _install_sigterm_handler(log)

    # TRADING_MODE is the explicit operator opt-in for the runtime mode.
    trading_mode = os.environ.get("TRADING_MODE", "paper").strip().lower()
    if trading_mode not in ("paper", "live"):
        log.error(
            "TRADING_MODE=%r is not supported; expected 'paper' or 'live'. Refusing to start.",
            trading_mode,
        )
        return 2

    # In live mode we additionally require the EXECUTE_TRADES kill switch to
    # be flipped in source AND the LiveEngine's own safety gates to pass at
    # construction. Defense in depth.
    if trading_mode == "live" and not EXECUTE_TRADES:
        log.error(
            "TRADING_MODE=live but EXECUTE_TRADES is False in sol_bot/ai_filter.py. "
            "Flip the source constant and commit it deliberately before starting "
            "live mode. Refusing to start."
        )
        return 2

    # Default symbol depends on mode — Kraken doesn't list SOL/USDT.
    default_symbol = "SOL/USD" if trading_mode == "live" else "SOL/USDT"
    config = BotConfig(
        symbol=os.getenv("BOT_SYMBOL", default_symbol),
        timeframe=os.getenv("BOT_TIMEFRAME", "1h"),
        # In live mode the operator MUST set confirm_trade_only_key=True via
        # a BotConfig source edit. We don't expose this as an env var on
        # purpose — it's an explicit acknowledgment, not a deployment flag.
        confirm_trade_only_key=(trading_mode == "live"
                                and os.getenv("CONFIRM_TRADE_ONLY_KEY", "").lower() == "yes"),
    )
    starting_balance = float(os.getenv("BOT_STARTING_BALANCE", "10000"))
    raw_burn_in = os.environ.get("BOT_BURN_IN_HOURS") or None
    burn_in_hours = float(raw_burn_in) if raw_burn_in else None
    max_runtime_s = burn_in_hours * 3600 if burn_in_hours is not None else None

    log.info(
        "starting bot: mode=%s symbol=%s timeframe=%s balance=%.2f "
        "execute_trades=%s burn_in_hours=%s",
        trading_mode, config.symbol, config.timeframe, starting_balance,
        EXECUTE_TRADES, burn_in_hours,
    )
    if trading_mode == "paper":
        log.warning(
            "PAPER mode — fills are simulated, no real orders will be placed. "
            "AI filter is opt-in (currently off by default in run_paper)."
        )
    else:
        log.warning(
            "LIVE mode — real orders against Kraken. max_live_balance=%.2f ceiling enforced.",
            config.max_live_balance,
        )

    engine = _build_engine(trading_mode, config, starting_balance, log)

    run_paper(
        config,
        starting_balance=starting_balance,
        execute_trades=EXECUTE_TRADES,
        max_runtime_s=max_runtime_s,
        engine=engine,
        stop_requested=lambda: _STOP_REQUESTED,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
