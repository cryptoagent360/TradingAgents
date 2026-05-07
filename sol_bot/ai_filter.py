"""Claude-powered binary trade filter for the Solana bot.

Runs **after** a valid strategy signal — never on its own. Returns
"APPROVE" or "REJECT". Every failure mode (cooldown active, request
timeout, API error, missing API key, malformed response) yields
"REJECT" — the filter fails closed, never opens a trade by accident.

EXECUTE_TRADES is the operator opt-in for real-money live execution.
It is a hardcoded source constant — flipping it to True requires an
explicit code edit and a git commit, leaving a review trail. The
runner consults this constant before placing live orders; paper mode
overrides it at the CLI layer because paper fills are simulated.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

import anthropic

logger = logging.getLogger(__name__)

EXECUTE_TRADES = False

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
        # Monotonic, not wall-clock: NTP step adjustments would otherwise make
        # the cooldown misfire (either gating us forever or letting a flood of
        # signals through if the clock jumps backwards).
        if time.monotonic() - _last_ai_call < AI_COOLDOWN_SECONDS:
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
            _last_ai_call = time.monotonic()
        # Defensive: a response with no text block (e.g. only thinking-summary
        # or tool_use) would StopIteration here, which would surface as a
        # confusing error inside the bare `except Exception` below. Treat
        # missing text as REJECT directly with a clearer log.
        text_blocks = [b for b in response.content if b.type == "text"]
        if not text_blocks:
            logger.warning("AI filter REJECT (no text block in response)")
            return "REJECT"
        decision = text_blocks[0].text.strip().upper()
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
