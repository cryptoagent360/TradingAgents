"""Lifecycle of an OpenTrade across a sequence of bars."""

import math

import pytest

from tradingagents.solana_bot.trade import OpenTrade

pytestmark = pytest.mark.unit


def _open_trade(entry=100.0, atr_value=2.0, size=10.0):
    return OpenTrade.open_long(
        entry=entry,
        atr_value=atr_value,
        size=size,
        atr_mult=1.5,
        tp1_r=1.0,
        tp2_r=2.0,
        tp1_close_fraction=0.5,
        tp2_close_fraction=0.25,
        trail_atr_mult=1.5,
    )


def test_open_long_levels_match_spec():
    t = _open_trade()
    # Stop: 100 - 1.5*2 = 97. R = 3. TP1 = 103. TP2 = 106.
    assert math.isclose(t.initial_stop, 97.0)
    assert math.isclose(t.tp1_price, 103.0)
    assert math.isclose(t.tp2_price, 106.0)
    assert math.isclose(t.tp1_size, 5.0)
    assert math.isclose(t.tp2_size, 2.5)
    assert math.isclose(t.runner_size, 2.5)


def test_tp1_partial_close_and_be_stop():
    t = _open_trade()
    fills = t.manage(high=103.5, low=100.0, close=103.2)
    assert len(fills) == 1
    assert fills[0].kind == "tp1"
    assert math.isclose(fills[0].price, 103.0)
    assert math.isclose(fills[0].size, 5.0)
    assert math.isclose(t.current_stop, 100.0)  # moved to BE
    assert math.isclose(t.remaining, 5.0)


def test_tp2_takes_another_quarter():
    t = _open_trade()
    t.manage(high=103.5, low=100.0, close=103.0)
    fills = t.manage(high=106.5, low=103.0, close=106.0)
    kinds = [f.kind for f in fills]
    assert "tp2" in kinds
    assert math.isclose(t.remaining, 2.5)


def test_full_stop_out_before_tp1():
    t = _open_trade()
    fills = t.manage(high=99.0, low=96.5, close=97.0)
    assert len(fills) == 1
    assert fills[0].kind == "stop"
    assert math.isclose(fills[0].size, 10.0)  # entire size closed
    assert t.is_closed()


def test_trailing_stop_advances_after_tp2():
    t = _open_trade()
    t.manage(high=103.5, low=100.0, close=103.0)
    t.manage(high=106.5, low=103.0, close=106.0)
    # Push higher; trail = 110 - 1.5*2 = 107.
    fills = t.manage(high=110.0, low=106.5, close=109.5)
    assert fills == []  # nothing exits; trail just advances
    assert math.isclose(t.current_stop, 107.0)


def test_trailing_stop_gets_hit_after_tp2():
    t = _open_trade()
    t.manage(high=103.5, low=100.0, close=103.0)
    t.manage(high=106.5, low=103.0, close=106.0)
    t.manage(high=110.0, low=106.5, close=109.5)
    fills = t.manage(high=109.0, low=106.5, close=107.0)
    assert any(f.kind == "trail" for f in fills)
    assert t.is_closed()


def test_realised_pnl_accounts_for_fees():
    t = _open_trade()
    fills = t.manage(high=103.5, low=100.0, close=103.0)
    pnl_no_fee = t.realised_pnl(fills, fee_rate=0.0)
    pnl_with_fee = t.realised_pnl(fills, fee_rate=0.001)
    assert pnl_with_fee < pnl_no_fee


def test_round_trip_serialization():
    t = _open_trade()
    t.manage(high=103.5, low=100.0, close=103.0)
    restored = OpenTrade.from_dict(t.to_dict())
    assert restored.current_stop == t.current_stop
    assert restored.remaining == t.remaining
    assert restored.tp1_hit == t.tp1_hit
