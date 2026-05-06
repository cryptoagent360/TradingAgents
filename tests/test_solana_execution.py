"""Execution engine tests."""

import math

import pytest

from tradingagents.solana_bot.config import BotConfig
from tradingagents.solana_bot.execution import LiveEngine, PaperEngine

pytestmark = pytest.mark.unit


def test_paper_engine_charges_entry_fee():
    cfg = BotConfig(taker_fee=0.001)
    engine = PaperEngine(cfg, balance=10_000)
    engine.open_long(entry_price=100.0, size=10.0, atr_value=2.0)
    # 100 * 10 * 0.001 = 1.0 fee.
    assert math.isclose(engine.balance, 10_000 - 1.0, rel_tol=1e-9)


def test_paper_engine_full_winner_increases_balance():
    cfg = BotConfig(taker_fee=0.0)
    engine = PaperEngine(cfg, balance=10_000)
    trade = engine.open_long(entry_price=100.0, size=10.0, atr_value=2.0)
    # Bar 1: TP1 hit at 103.
    r1 = engine.manage(trade, high=103.5, low=100.0, close=103.0)
    assert r1.realised_pnl > 0
    # Bar 2: TP2 hit at 106.
    r2 = engine.manage(trade, high=106.5, low=103.0, close=106.0)
    assert r2.realised_pnl > 0
    # Total balance should reflect both partial exits.
    assert engine.balance > 10_000


def test_live_engine_is_stubbed():
    with pytest.raises(NotImplementedError) as exc:
        LiveEngine()
    assert "next PR" in str(exc.value) or "stubbed" in str(exc.value)
