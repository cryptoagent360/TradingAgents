"""Execution engine tests — PaperEngine + LiveEngine with mocked Kraken."""

import math
from unittest.mock import MagicMock

import pytest

from tradingagents.solana_bot.config import BotConfig
from tradingagents.solana_bot.execution import (
    BelowMinNotional,
    CredentialsRejected,
    LiveBalanceTooLarge,
    LiveEngine,
    PaperEngine,
    PaperFillReport,
    ReconcileReport,
    StopPlacementAndFlattenFailed,
    StopPlacementFailedFlattened,
    TradeOnlyKeyNotConfirmed,
    _floor_to_step,
)

pytestmark = pytest.mark.unit


# ------------------------- PaperEngine --------------------------------------


def test_paper_engine_charges_entry_fee():
    cfg = BotConfig(taker_fee=0.001)
    engine = PaperEngine(cfg, balance=10_000)
    engine.open_long(entry_price=100.0, size=10.0, atr_value=2.0)
    assert math.isclose(engine.balance, 10_000 - 1.0, rel_tol=1e-9)


def test_paper_engine_full_winner_increases_balance():
    cfg = BotConfig(taker_fee=0.0)
    engine = PaperEngine(cfg, balance=10_000)
    trade = engine.open_long(entry_price=100.0, size=10.0, atr_value=2.0)
    r1 = engine.manage(trade, high=103.5, low=100.0, close=103.0)
    assert r1.realised_pnl > 0
    r2 = engine.manage(trade, high=106.5, low=103.0, close=106.0)
    assert r2.realised_pnl > 0
    assert engine.balance > 10_000


def test_paper_engine_reconcile_returns_local_state():
    cfg = BotConfig()
    engine = PaperEngine(cfg, balance=12_345.0)
    report = engine.reconcile()
    assert isinstance(report, ReconcileReport)
    assert report.quote_balance == 12_345.0
    assert report.base_balance == 0.0
    assert report.open_orders == []
    assert report.is_live is False


def test_floor_to_step_rounds_down_never_up():
    assert _floor_to_step(0.123456, 0.01) == 0.12
    assert _floor_to_step(99.9999, 0.001) == 99.999
    assert _floor_to_step(5.0, 1.0) == 5.0
    assert _floor_to_step(3.14159, 0.0) == 3.14159


# ------------------------- LiveEngine helpers -------------------------------


def _safe_client(quote_balance: float = 50.0, open_orders: list | None = None) -> MagicMock:
    """Build a fake ccxt-kraken client with a known-safe account."""
    client = MagicMock()
    client.fetch_balance.return_value = {
        "USD": {"free": quote_balance, "used": 0.0, "total": quote_balance},
        "SOL": {"free": 0.0, "used": 0.0, "total": 0.0},
    }
    client.fetch_open_orders.return_value = open_orders or []
    client.load_markets.return_value = {
        "SOL/USD": {
            "precision": {"amount": 0.01, "price": 0.001},
            "limits": {
                "amount": {"min": 0.01},
                "price": {"min": 0.001},
                "cost": {"min": 5.0},
            },
        },
    }
    client.create_order.side_effect = lambda **kw: {
        "id": f"exch-{kw.get('params', {}).get('clientOrderId', 'unk')}",
        "average": kw.get("price", 100.0),
        "info": kw,
    }
    client.cancel_order.return_value = {"id": "cancelled"}
    return client


def _live_cfg(**overrides):
    base = dict(
        symbol="SOL/USD", exchange="kraken",
        max_live_balance=100.0, confirm_trade_only_key=True,
    )
    base.update(overrides)
    return BotConfig(**base)


# ------------------------- LiveEngine constructor / safety ------------------


def test_live_engine_refuses_without_trade_only_acknowledgment():
    cfg = BotConfig(symbol="SOL/USD", exchange="kraken", confirm_trade_only_key=False)
    with pytest.raises(TradeOnlyKeyNotConfirmed):
        LiveEngine(cfg, api_key="k", api_secret="s", client=_safe_client())


def test_live_engine_refuses_when_credentials_fail_smoke_test():
    cfg = _live_cfg()
    bad_client = _safe_client()
    bad_client.fetch_balance.side_effect = RuntimeError("EAPI:Invalid key")
    with pytest.raises(CredentialsRejected):
        LiveEngine(cfg, api_key="k", api_secret="s", client=bad_client)


def test_live_engine_refuses_balance_above_ceiling():
    cfg = _live_cfg()
    with pytest.raises(LiveBalanceTooLarge):
        LiveEngine(cfg, api_key="k", api_secret="s", client=_safe_client(quote_balance=500.0))


def test_live_engine_constructs_when_all_gates_pass():
    cfg = _live_cfg()
    engine = LiveEngine(cfg, api_key="k", api_secret="s", client=_safe_client(50.0))
    report = engine.reconcile()
    assert report.is_live is True
    assert report.quote_balance == 50.0


# ------------------------- LiveEngine open_long happy path ------------------


def test_open_long_places_entry_stop_tp1_tp2_with_deterministic_coids():
    cfg = _live_cfg()
    client = _safe_client(50.0)
    engine = LiveEngine(cfg, api_key="k", api_secret="s", client=client)

    trade = engine.open_long(entry_price=100.0, size=0.1, atr_value=2.0, trade_id="abc12345")

    # Four orders: market entry, stop, tp1 limit, tp2 limit
    assert client.create_order.call_count == 4
    types = [c.kwargs["type"] for c in client.create_order.call_args_list]
    assert types == ["market", "stop-loss", "limit", "limit"]

    coids = [c.kwargs["params"]["clientOrderId"] for c in client.create_order.call_args_list]
    assert coids == ["sb-abc12345-entry", "sb-abc12345-stop", "sb-abc12345-tp1", "sb-abc12345-tp2"]

    # All IDs are surfaced for runner persistence
    assert engine.last_order_ids["trade_id"] == "abc12345"
    assert engine.last_order_ids["stop"] == "sb-abc12345-stop"
    assert trade.entry == 100.0  # mock returned average=price for the market order


def test_open_long_rounds_size_down_to_lot_step():
    cfg = _live_cfg()
    client = _safe_client(50.0)
    engine = LiveEngine(cfg, api_key="k", api_secret="s", client=client)
    engine.open_long(entry_price=100.0, size=0.123, atr_value=2.0)
    # First create_order call is the market entry; check the rounded amount
    entry_call = client.create_order.call_args_list[0]
    assert entry_call.kwargs["amount"] == 0.12  # 0.123 floored to step 0.01


def test_open_long_refuses_below_min_notional():
    cfg = _live_cfg()
    client = _safe_client(50.0)
    engine = LiveEngine(cfg, api_key="k", api_secret="s", client=client)
    # 0.01 SOL * $100 = $1, below the $5 min_notional in the fixture
    with pytest.raises(BelowMinNotional):
        engine.open_long(entry_price=100.0, size=0.01, atr_value=2.0)


def test_open_long_uses_actual_fill_price_for_R_math():
    """Slippage: planned entry was $100, market fill came back at $101.
    R, stop, TP1, TP2 must be computed from $101, not $100."""
    cfg = _live_cfg()
    client = _safe_client(50.0)
    # Override create_order to return an "average" different from the requested price
    def _slip_create(**kw):
        if kw["type"] == "market":
            return {"id": "exch-entry", "average": 101.0, "info": kw}
        return {"id": f"exch-{kw['params']['clientOrderId']}", "info": kw}
    client.create_order.side_effect = _slip_create
    engine = LiveEngine(cfg, api_key="k", api_secret="s", client=client)

    trade = engine.open_long(entry_price=100.0, size=0.1, atr_value=2.0, trade_id="slip0001")

    # actual_entry was 101.0 → stop = 101 - 1.5*2 = 98.0, R = 3.0
    # tp1 = 101 + 1*3 = 104, tp2 = 101 + 2*3 = 107
    stop_call = client.create_order.call_args_list[1].kwargs
    assert stop_call["params"]["stopPrice"] == 98.0
    tp1_call = client.create_order.call_args_list[2].kwargs
    assert tp1_call["price"] == 104.0
    tp2_call = client.create_order.call_args_list[3].kwargs
    assert tp2_call["price"] == 107.0


# ------------------------- Stop-failure flatten ------------------------------


def test_open_long_flattens_position_when_stop_placement_fails():
    """If the stop fails to place after the entry filled, market-sell the
    position immediately. Naked-long exposure window must be closed."""
    cfg = _live_cfg()
    client = _safe_client(50.0)
    calls_made = []
    def _create(**kw):
        calls_made.append(kw)
        if kw["type"] == "stop-loss":
            raise RuntimeError("EOrder:Insufficient margin")
        return {"id": f"exch-{kw['params']['clientOrderId']}", "average": 100.0, "info": kw}
    client.create_order.side_effect = _create

    engine = LiveEngine(cfg, api_key="k", api_secret="s", client=client)
    with pytest.raises(StopPlacementFailedFlattened):
        engine.open_long(entry_price=100.0, size=0.1, atr_value=2.0, trade_id="flat0001")

    # Three orders attempted: entry (filled), stop (failed), flatten (filled)
    types = [c["type"] for c in calls_made]
    assert types == ["market", "stop-loss", "market"]
    # The flatten call sells the same size
    flatten_call = calls_made[2]
    assert flatten_call["side"] == "sell"
    assert flatten_call["amount"] == 0.1
    assert "flatten" in flatten_call["params"]["clientOrderId"]


def test_open_long_raises_catastrophic_if_flatten_also_fails():
    """Entry filled, stop failed, AND flatten failed — operator MUST intervene."""
    cfg = _live_cfg()
    client = _safe_client(50.0)
    def _create(**kw):
        if kw["type"] == "stop-loss":
            raise RuntimeError("stop failed")
        if kw["type"] == "market" and kw["side"] == "sell":
            raise RuntimeError("flatten failed")
        return {"id": f"exch-{kw['params']['clientOrderId']}", "average": 100.0, "info": kw}
    client.create_order.side_effect = _create

    engine = LiveEngine(cfg, api_key="k", api_secret="s", client=client)
    with pytest.raises(StopPlacementAndFlattenFailed):
        engine.open_long(entry_price=100.0, size=0.1, atr_value=2.0)


def test_open_long_continues_if_tp1_or_tp2_placement_fails():
    """TP placements are best-effort — the stop is in place, so the position
    is still safe. We log and continue."""
    cfg = _live_cfg()
    client = _safe_client(50.0)
    def _create(**kw):
        if kw["type"] == "limit":
            raise RuntimeError("EOrder:Rate limit")
        return {"id": f"exch-{kw['params']['clientOrderId']}", "average": 100.0, "info": kw}
    client.create_order.side_effect = _create

    engine = LiveEngine(cfg, api_key="k", api_secret="s", client=client)
    # Does NOT raise — TPs are best-effort
    trade = engine.open_long(entry_price=100.0, size=0.1, atr_value=2.0, trade_id="tp_fail1")
    assert trade is not None
    assert engine._active_orders["tp1"] is None
    assert engine._active_orders["tp2"] is None
    assert engine._active_orders["stop"] is not None


# ------------------------- Per-trade balance recheck ------------------------


def test_open_long_rechecks_balance_ceiling_per_trade():
    """Construction passed at $50, but a mid-run deposit pushed balance to $200.
    The next open_long must refuse."""
    cfg = _live_cfg()
    client = _safe_client(50.0)
    engine = LiveEngine(cfg, api_key="k", api_secret="s", client=client)
    # Simulate a deposit between construction and the next trade
    client.fetch_balance.return_value = {
        "USD": {"free": 200.0, "used": 0.0, "total": 200.0},
        "SOL": {"free": 0.0, "used": 0.0, "total": 0.0},
    }
    with pytest.raises(LiveBalanceTooLarge):
        engine.open_long(entry_price=100.0, size=0.1, atr_value=2.0)


# ------------------------- manage() poll-based fills -----------------------


def test_manage_emits_fill_event_when_tp1_order_no_longer_open():
    cfg = _live_cfg()
    client = _safe_client(50.0)
    engine = LiveEngine(cfg, api_key="k", api_secret="s", client=client)
    trade = engine.open_long(entry_price=100.0, size=0.1, atr_value=2.0, trade_id="mng00001")

    # Initially three open orders: stop, tp1, tp2
    initial = [
        {"clientOrderId": "sb-mng00001-stop"},
        {"clientOrderId": "sb-mng00001-tp1"},
        {"clientOrderId": "sb-mng00001-tp2"},
    ]
    # Simulate TP1 filling: it disappears from open_orders
    client.fetch_open_orders.return_value = [
        {"clientOrderId": "sb-mng00001-stop"},
        {"clientOrderId": "sb-mng00001-tp2"},
    ]
    report = engine.manage(trade, high=105.0, low=99.0, close=103.0)
    assert len(report.fills) == 1
    assert report.fills[0].kind == "tp1"
    assert trade.tp1_hit is True


def test_manage_returns_empty_when_no_orders_filled():
    cfg = _live_cfg()
    client = _safe_client(50.0)
    engine = LiveEngine(cfg, api_key="k", api_secret="s", client=client)
    trade = engine.open_long(entry_price=100.0, size=0.1, atr_value=2.0, trade_id="mng00002")
    # All orders still resting
    client.fetch_open_orders.return_value = [
        {"clientOrderId": "sb-mng00002-stop"},
        {"clientOrderId": "sb-mng00002-tp1"},
        {"clientOrderId": "sb-mng00002-tp2"},
    ]
    report = engine.manage(trade, high=101.0, low=99.5, close=100.5)
    assert report.fills == []
    assert report.realised_pnl == 0.0


def test_manage_handles_fetch_open_orders_failure_gracefully():
    """A failed poll should not crash the runner — return empty, retry next cycle."""
    cfg = _live_cfg()
    client = _safe_client(50.0)
    engine = LiveEngine(cfg, api_key="k", api_secret="s", client=client)
    trade = engine.open_long(entry_price=100.0, size=0.1, atr_value=2.0)
    client.fetch_open_orders.side_effect = RuntimeError("network")
    report = engine.manage(trade, high=101.0, low=99.5, close=100.5)
    assert isinstance(report, PaperFillReport)
    assert report.fills == []
