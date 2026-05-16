"""Position-sizing math and the daily-loss kill switch."""

import math
from datetime import date, timedelta
from unittest.mock import patch

import pytest

from tradingagents.solana_bot.risk import DailyLossTracker, position_size

pytestmark = pytest.mark.unit


def test_position_size_known_values():
    # 1% of $10,000 = $100 risk. Risk distance = $1.50. Size = 100 / 1.5 ≈ 66.667.
    size = position_size(balance=10_000, entry=100.0, stop=98.5, risk_pct=0.01)
    assert math.isclose(size, 100 / 1.5, rel_tol=1e-9)


def test_position_size_rejects_inverted_stop():
    with pytest.raises(ValueError):
        position_size(balance=10_000, entry=100.0, stop=101.0, risk_pct=0.01)


def test_position_size_rejects_zero_balance():
    with pytest.raises(ValueError):
        position_size(balance=0, entry=100.0, stop=99.0, risk_pct=0.01)


def test_position_size_rejects_bad_risk_pct():
    with pytest.raises(ValueError):
        position_size(balance=10_000, entry=100.0, stop=99.0, risk_pct=0.0)
    with pytest.raises(ValueError):
        position_size(balance=10_000, entry=100.0, stop=99.0, risk_pct=1.0)


def test_kill_switch_trips_at_threshold():
    tracker = DailyLossTracker(starting_balance=10_000, max_daily_loss_pct=0.03)
    tracker.record_pnl(-200)
    assert tracker.can_open_trade()
    tracker.record_pnl(-100)  # cumulative -300, exactly the threshold
    assert not tracker.can_open_trade()


def test_kill_switch_does_not_trip_before_threshold():
    tracker = DailyLossTracker(starting_balance=10_000, max_daily_loss_pct=0.03)
    tracker.record_pnl(-200)
    tracker.record_pnl(-99)
    assert tracker.can_open_trade()


def test_utc_rollover_resets_pnl_and_kill_switch():
    today = date(2026, 5, 6)
    tomorrow = today + timedelta(days=1)

    with patch("tradingagents.solana_bot.risk.datetime") as mock_dt:
        mock_dt.now.return_value.date.return_value = today
        tracker = DailyLossTracker(starting_balance=10_000, max_daily_loss_pct=0.03)
        tracker.record_pnl(-500)
        assert tracker.kill_switch_triggered
        assert not tracker.can_open_trade()

    with patch("tradingagents.solana_bot.risk.datetime") as mock_dt:
        mock_dt.now.return_value.date.return_value = tomorrow
        # Trigger a rollover via any tracker access.
        assert tracker.can_open_trade()
        assert tracker.today_pnl == 0.0
        assert not tracker.kill_switch_triggered


def test_round_trip_serialization():
    tracker = DailyLossTracker(starting_balance=10_000, max_daily_loss_pct=0.03)
    tracker.record_pnl(-150)
    restored = DailyLossTracker.from_dict(tracker.to_dict())
    assert restored.today_pnl == tracker.today_pnl
    assert restored.kill_switch_triggered == tracker.kill_switch_triggered
    assert restored.starting_balance == tracker.starting_balance
