"""Trend + pullback Solana trading bot.

Rule-based, no LLM in the hot loop. Ships with a custom event-driven
backtester and a paper-trade runner. Live execution is intentionally
stubbed in this release; see ``LiveEngine``.
"""

from tradingagents.solana_bot.config import BotConfig

__all__ = ["BotConfig"]
