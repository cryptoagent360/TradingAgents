"""Execution engine tests."""

import math
from unittest.mock import MagicMock

import pytest

from tradingagents.solana_bot.config import BotConfig
from tradingagents.solana_bot.execution import (
    BelowMinNotional,
    LiveBalanceTooLarge,
    LiveEngine,
    PaperEngine,
    ReconcileReport,
    WithdrawPermissionEnabled,
    _floor_to_step,
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


def _safe_client(
    quote_balance: float = 50.0,
    *,
    withdraw: bool = False,
    transfer: bool = False,
) -> MagicMock:
    """Build a fake ccxt client with a known-safe account."""
    client = MagicMock()
    client.fetch_account_permissions.return_value = {
        "withdraw": withdraw, "transfer": transfer, "trade": True,
    }
    client.fetch_balance.return_value = {
        "USDT": {"free": quote_balance, "used": 0.0, "total": quote_balance},
        "SOL": {"free": 0.0, "used": 0.0, "total": 0.0},
    }
    client.fetch_open_orders.return_value = []
    # Realistic Binance SOL/USDT-style market metadata.
    client.load_markets.return_value = {
        "SOL/USDT": {
            "precision": {"amount": 0.01, "price": 0.001},
            "limits": {
                "amount": {"min": 0.01},
                "price": {"min": 0.001},
                "cost": {"min": 5.0},
            },
        },
    }
    # create_order returns whatever the exchange returns; tests don't care about shape
    # except for the exchange order id field.
    client.create_order.side_effect = lambda **kwargs: {
        "id": f"exch-{kwargs.get('params', {}).get('newClientOrderId', 'unk')}",
        "info": kwargs,
    }
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


def test_live_engine_refuses_internal_transfer_permission():
    """Internal-transfer can move funds to another Binance account — also refuse."""
    cfg = BotConfig(max_live_balance=100.0)
    with pytest.raises(WithdrawPermissionEnabled):
        LiveEngine(cfg, api_key="k", api_secret="s", testnet=True, client=_safe_client(transfer=True))


def test_live_engine_refuses_when_permissions_response_is_empty():
    """Empty/unexpected permission shape used to default-allow; now defaults to refuse."""
    cfg = BotConfig(max_live_balance=100.0)
    client = _safe_client(50.0)
    client.fetch_account_permissions.return_value = {}  # exchange returned nothing useful
    with pytest.raises(WithdrawPermissionEnabled):
        LiveEngine(cfg, api_key="k", api_secret="s", testnet=True, client=client)


def test_live_engine_refuses_balance_above_ceiling():
    cfg = BotConfig(max_live_balance=100.0)
    with pytest.raises(LiveBalanceTooLarge):
        LiveEngine(cfg, api_key="k", api_secret="s", testnet=True, client=_safe_client(quote_balance=500.0))


def test_floor_to_step_rounds_down_never_up():
    assert _floor_to_step(0.123456, 0.01) == 0.12
    assert _floor_to_step(99.9999, 0.001) == 99.999
    assert _floor_to_step(5.0, 1.0) == 5.0
    # Step of 0 (or negative) is a degenerate "no rounding".
    assert _floor_to_step(3.14159, 0.0) == 3.14159


def test_live_engine_open_long_places_market_entry_and_stop_loss():
    cfg = BotConfig(max_live_balance=100.0, atr_mult=1.5)
    client = _safe_client(50.0)
    # Make the entry order report a fill price to exercise the actual_entry path.
    client.create_order.side_effect = lambda **kwargs: {
        "id": f"exch-{kwargs.get('params', {}).get('newClientOrderId', 'unk')}",
        "average": 100.0,
        "info": kwargs,
    }
    engine = LiveEngine(cfg, api_key="k", api_secret="s", testnet=True, client=client)

    trade = engine.open_long(entry_price=100.0, size=0.1, atr_value=2.0)

    # Two orders placed: market entry + protective stop.
    assert client.create_order.call_count == 2
    calls = client.create_order.call_args_list
    entry_call = calls[0].kwargs
    stop_call = calls[1].kwargs

    # Entry: market buy.
    assert entry_call["type"] == "market"
    assert entry_call["side"] == "buy"
    assert entry_call["amount"] == 0.1
    assert entry_call["params"]["newClientOrderId"].startswith("sb-entry-")

    # Stop: STOP_LOSS_LIMIT sell at entry - atr_mult * atr = 100 - 1.5*2 = 97.0.
    assert stop_call["type"] == "STOP_LOSS_LIMIT"
    assert stop_call["side"] == "sell"
    assert stop_call["amount"] == 0.1
    assert stop_call["price"] == 97.0
    assert stop_call["params"]["stopPrice"] == 97.0
    assert stop_call["params"]["newClientOrderId"].startswith("sb-stop-")

    # Trade is well-formed; order IDs are surfaced for the runner to persist.
    assert trade.entry == 100.0
    assert trade.size == 0.1
    assert engine.last_order_ids["entry"].startswith("sb-entry-")
    assert engine.last_order_ids["stop"].startswith("sb-stop-")


def test_live_engine_open_long_rounds_size_to_market_lot_step():
    cfg = BotConfig(max_live_balance=100.0)
    client = _safe_client(50.0)
    engine = LiveEngine(cfg, api_key="k", api_secret="s", testnet=True, client=client)

    # 0.123 with lot step 0.01 must floor to 0.12.
    engine.open_long(entry_price=100.0, size=0.123, atr_value=2.0)
    assert client.create_order.call_args_list[0].kwargs["amount"] == 0.12


def test_live_engine_open_long_refuses_below_min_notional():
    cfg = BotConfig(max_live_balance=100.0)
    client = _safe_client(50.0)
    engine = LiveEngine(cfg, api_key="k", api_secret="s", testnet=True, client=client)

    # 0.01 SOL * $100 = $1 — well below the $5 min_notional in the fixture.
    with pytest.raises(BelowMinNotional):
        engine.open_long(entry_price=100.0, size=0.01, atr_value=2.0)
    # No orders placed.
    assert client.create_order.call_count == 0


def test_live_engine_uses_distinct_idempotent_order_ids():
    cfg = BotConfig(max_live_balance=100.0)
    client = _safe_client(50.0)
    engine = LiveEngine(cfg, api_key="k", api_secret="s", testnet=True, client=client)

    engine.open_long(entry_price=100.0, size=0.1, atr_value=2.0)
    first = dict(engine.last_order_ids)
    engine.open_long(entry_price=100.0, size=0.1, atr_value=2.0)
    second = dict(engine.last_order_ids)
    # Each call gets a fresh client_order_id — re-submitting the same one would
    # be a no-op on Binance (idempotency guarantee), but new trades need new IDs.
    assert first["entry"] != second["entry"]
    assert first["stop"] != second["stop"]


def test_live_engine_manage_still_raises_until_orders_land():
    cfg = BotConfig(max_live_balance=100.0)
    engine = LiveEngine(cfg, api_key="k", api_secret="s", testnet=True, client=_safe_client(50.0))
    trade = engine.open_long(entry_price=100.0, size=0.1, atr_value=2.0)
    with pytest.raises(NotImplementedError):
        engine.manage(trade, high=105.0, low=99.0, close=104.0)


def test_live_engine_open_long_uses_actual_fill_price_not_planned():
    """Slippage matters — OpenTrade.entry must be the real fill, not the planned one."""
    cfg = BotConfig(max_live_balance=100.0, atr_mult=1.5)
    client = _safe_client(50.0)
    # Market filled $0.50 above the planned price (slippage on a fast move).
    client.create_order.side_effect = lambda **kwargs: {
        "id": "exch-x",
        "average": 100.5 if kwargs["type"] == "market" else None,
        "info": kwargs,
    }
    engine = LiveEngine(cfg, api_key="k", api_secret="s", testnet=True, client=client)

    trade = engine.open_long(entry_price=100.0, size=0.1, atr_value=2.0)
    assert trade.entry == 100.5  # actual fill, not planned 100.0


def test_live_engine_rechecks_balance_on_every_open():
    """A mid-run deposit that bursts the ceiling must be caught at the next trade."""
    cfg = BotConfig(max_live_balance=100.0)
    client = _safe_client(50.0)
    engine = LiveEngine(cfg, api_key="k", api_secret="s", testnet=True, client=client)
    # Construction passed; now simulate a deposit before the first trade.
    client.fetch_balance.return_value = {
        "USDT": {"free": 5_000.0, "used": 0.0, "total": 5_000.0},
        "SOL": {"free": 0.0, "used": 0.0, "total": 0.0},
    }
    with pytest.raises(LiveBalanceTooLarge):
        engine.open_long(entry_price=100.0, size=0.1, atr_value=2.0)


def test_live_engine_caches_market_metadata():
    """load_markets() must be called at most once per engine instance."""
    cfg = BotConfig(max_live_balance=100.0)
    client = _safe_client(50.0)
    engine = LiveEngine(cfg, api_key="k", api_secret="s", testnet=True, client=client)
    engine.open_long(entry_price=100.0, size=0.1, atr_value=2.0)
    # _round_amount + _round_price (twice: entry + stop) + _check_min_notional
    # would have called load_markets() 4 times if not cached.
    assert client.load_markets.call_count == 1


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
