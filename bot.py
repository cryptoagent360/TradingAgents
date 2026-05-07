"""Long-running entry point for the Solana trend+pullback bot.

Suitable for background deployment via nohup or systemd:

    nohup python bot.py > output.log 2>&1 &

Loads .env at startup so ANTHROPIC_API_KEY (for the AI filter) and any
exchange credentials are available. Honors the EXECUTE_TRADES gate
defined in sol_bot/ai_filter.py verbatim — flip it to True in source
to enable real-money execution.

Operator-tunable knobs are read from the environment with sensible
defaults (no CLI flags, since this is meant to run unattended):

    BOT_STARTING_BALANCE  account size in quote currency (default 10000)
    BOT_SYMBOL            trading pair (default SOL/USDT)
    BOT_TIMEFRAME         candle timeframe (default 1h)
    BOT_BURN_IN_HOURS     wall-clock budget; loop exits after N hours (default unlimited)
"""

from __future__ import annotations

import logging
import os
import sys

from dotenv import load_dotenv

from sol_bot.ai_filter import EXECUTE_TRADES
from tradingagents.solana_bot.config import BotConfig
from tradingagents.solana_bot.runner import run_paper


def main() -> int:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    log = logging.getLogger(__name__)

    config = BotConfig(
        symbol=os.getenv("BOT_SYMBOL", "SOL/USDT"),
        timeframe=os.getenv("BOT_TIMEFRAME", "1h"),
    )
    starting_balance = float(os.getenv("BOT_STARTING_BALANCE", "10000"))
    burn_in_hours = float(os.environ["BOT_BURN_IN_HOURS"]) if "BOT_BURN_IN_HOURS" in os.environ else None
    max_runtime_s = burn_in_hours * 3600 if burn_in_hours is not None else None

    log.info(
        "starting bot: symbol=%s timeframe=%s balance=%.2f execute_trades=%s burn_in_hours=%s",
        config.symbol,
        config.timeframe,
        starting_balance,
        EXECUTE_TRADES,
        burn_in_hours,
    )
    if not EXECUTE_TRADES:
        log.warning(
            "EXECUTE_TRADES is False — signals and AI decisions will be logged "
            "but no orders will be placed. Edit sol_bot/ai_filter.py to enable."
        )

    run_paper(
        config,
        starting_balance=starting_balance,
        execute_trades=EXECUTE_TRADES,
        max_runtime_s=max_runtime_s,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
