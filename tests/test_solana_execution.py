"""Execution engine tests."""

import math
from unittest.mock import MagicMock

import pytest

from tradingagents.solana_bot.config import BotConfig
from tradingagents.solana_bot.execution import (
    LiveBalanceTooLarge,
    LiveEngine,
    PaperEngine,
    ReconcileReport,
    WithdrawPermissionEnabled,
)

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


def test_paper_engine_reconcile_returns_local_state():
    cfg = BotConfig()
    engine = PaperEngine(cfg, balance=12_345.0)
    report = engine.reconcile()
    assert isinstance(report, ReconcileReport)
    assert report.quote_balance == 12_345.0
    assert report.base_balance == 0.0
    assert report.open_orders == []


def _safe_client(quote_balance: float = 50.0, withdraw: bool = False) -> MagicMock:
    """Build a fake ccxt client with a known-safe account."""
    client = MagicMock()
    client.fetch_account_permissions.return_value = {"withdraw": withdraw, "trade": True}
    client.fetch_balance.return_value = {
        "USDT": {"free": quote_balance, "used": 0.0, "total": quote_balance},
        "SOL": {"free": 0.0, "used": 0.0, "total": 0.0},
    }
    client.fetch_open_orders.return_value = []
    return client


def test_live_engine_constructs_with_safe_account():
    cfg = BotConfig(max_live_balance=100.0)
    engine = LiveEngine(cfg, api_key="k", api_secret="s", testnet=True, client=_safe_client(50.0))
    report = engine.reconcile()
    assert report.quote_balance == 50.0


def test_live_engine_refuses_withdraw_permission():
    cfg = BotConfig(max_live_balance=100.0)
    with pytest.raises(WithdrawPermissionEnabled):
        LiveEngine(cfg, api_key="k", api_secret="s", testnet=True, client=_safe_client(withdraw=True))


def test_live_engine_refuses_balance_above_ceiling():
    cfg = BotConfig(max_live_balance=100.0)
    with pytest.raises(LiveBalanceTooLarge):
        LiveEngine(cfg, api_key="k", api_secret="s", testnet=True, client=_safe_client(quote_balance=500.0))


def test_live_engine_open_long_still_raises_until_orders_land():
    cfg = BotConfig(max_live_balance=100.0)
    engine = LiveEngine(cfg, api_key="k", api_secret="s", testnet=True, client=_safe_client(50.0))
    with pytest.raises(NotImplementedError) as exc:
        engine.open_long(entry_price=100.0, size=0.1, atr_value=2.0)
    assert "next commit" in str(exc.value)


def test_live_engine_reconcile_returns_balances_and_open_orders():
    cfg = BotConfig(max_live_balance=100.0)
    client = _safe_client(50.0)
    client.fetch_balance.return_value = {
        "USDT": {"free": 50.0, "used": 0.0, "total": 50.0},
        "SOL": {"free": 1.5, "used": 0.0, "total": 1.5},
    }
    client.fetch_open_orders.return_value = [{"id": "abc", "side": "sell", "price": 110.0}]
    engine = LiveEngine(cfg, api_key="k", api_secret="s", testnet=True, client=client)
    report = engine.reconcile()
    assert report.quote_balance == 50.0
    assert report.base_balance == 1.5
    assert report.open_orders == [{"id": "abc", "side": "sell", "price": 110.0}]
