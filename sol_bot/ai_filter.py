"""Claude-powered binary trade filter for the Solana bot.

Runs **after** a valid strategy signal — never on its own. Returns
"APPROVE" or "REJECT". Every failure mode (cooldown active, request
timeout, API error, missing API key, malformed response) yields
"REJECT" — the filter fails closed, never opens a trade by accident.

EXECUTE_TRADES is the operator opt-in for real-money execution. It is
read once from the environment at module import:

    EXECUTE_TRADES=1 tradingagents solana paper ...

Default is False. The runner consults this constant before placing any
order; this is the *only* execution-gate path.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

import anthropic

logger = logging.getLogger(__name__)

EXECUTE_TRADES: bool = os.getenv("EXECUTE_TRADES", "").lower() in ("1", "true", "yes")

AI_COOLDOWN_SECONDS = 60.0
AI_TIMEOUT_SECONDS = 5.0
AI_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = (
    "You are a strict crypto trading risk filter. "
    "Given a setup, respond with exactly one word: APPROVE or REJECT. "
    "No explanation, no punctuation."
)

_client_lock = threading.Lock()
_client: Optional[anthropic.Anthropic] = None
_last_call_lock = threading.Lock()
_last_ai_call: float = 0.0


def _get_client() -> anthropic.Anthropic:
    global _client
    with _client_lock:
        if _client is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set")
            _client = anthropic.Anthropic(api_key=api_key, timeout=AI_TIMEOUT_SECONDS)
    return _client


def ai_trade_filter(market_data: dict) -> str:
    global _last_ai_call

    with _last_call_lock:
        if time.time() - _last_ai_call < AI_COOLDOWN_SECONDS:
            logger.info("AI filter REJECT (cooldown active)")
            return "REJECT"

    user_message = (
        "Evaluate this SOL setup:\n\n"
        f"Trend: {market_data['trend']}\n"
        f"RSI: {market_data['rsi']}\n"
        f"Distance from EMA50: {market_data['distance']}\n"
        f"Volume condition: {market_data['volume']}"
    )

    try:
        start = time.time()
        response = _get_client().messages.create(
            model=AI_MODEL,
            max_tokens=5,
            thinking={"type": "disabled"},
            output_config={"effort": "low"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        with _last_call_lock:
            _last_ai_call = time.time()
        decision = next(
            block.text for block in response.content if block.type == "text"
        ).strip().upper()
        latency = time.time() - start
        logger.info("AI filter %s (latency=%.2fs)", decision, latency)
        return "APPROVE" if decision == "APPROVE" else "REJECT"
    except anthropic.APITimeoutError:
        logger.warning("AI filter REJECT (timeout)")
        return "REJECT"
    except anthropic.APIError as exc:
        logger.warning("AI filter REJECT (API error: %s)", exc)
        return "REJECT"
    except Exception as exc:
        logger.warning("AI filter REJECT (unexpected error: %s)", exc)
        return "REJECT"
