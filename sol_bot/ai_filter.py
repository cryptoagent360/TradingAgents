import os
import time

import anthropic

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

EXECUTE_TRADES = False
AI_COOLDOWN_SECONDS = 60

_last_ai_call = 0.0

SYSTEM_PROMPT = (
    "You are a strict crypto trading risk filter. "
    "Given a setup, respond with exactly one word: APPROVE or REJECT. "
    "No explanation, no punctuation."
)


def ai_trade_filter(market_data):
    global _last_ai_call

    if time.time() - _last_ai_call < AI_COOLDOWN_SECONDS:
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

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=5,
            thinking={"type": "disabled"},
            output_config={"effort": "low"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        _last_ai_call = time.time()

        decision = next(
            block.text for block in response.content if block.type == "text"
        ).strip()

        latency = time.time() - start
        print(f"AI latency: {latency:.2f}s | Decision: {decision}")

        return decision

    except anthropic.APIError as e:
        print(f"AI ERROR: {e}")
        return "REJECT"


def execute_trade():
    raise NotImplementedError(
        "Live execution stub — wire ccxt order placement here before flipping "
        "EXECUTE_TRADES = True."
    )


def dispatch_trade():
    if EXECUTE_TRADES:
        execute_trade()
    else:
        print("SIMULATION: Trade would execute")
