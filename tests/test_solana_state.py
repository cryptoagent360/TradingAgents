"""Atomic JSON persistence for the bot state."""

import json

import pytest

from tradingagents.solana_bot.risk import DailyLossTracker
from tradingagents.solana_bot.state import STATE_VERSION, BotState, StateVersionMismatch
from tradingagents.solana_bot.trade import OpenTrade

pytestmark = pytest.mark.unit


def _open_trade():
    return OpenTrade.open_long(
        entry=100.0,
        atr_value=2.0,
        size=10.0,
        atr_mult=1.5,
        tp1_r=1.0,
        tp2_r=2.0,
        tp1_close_fraction=0.5,
        tp2_close_fraction=0.25,
        trail_atr_mult=1.5,
    )


def test_save_then_load_round_trip(tmp_path):
    path = tmp_path / "state.json"
    state = BotState(path=path, open_trade=_open_trade(), last_candle_ts=42)
    state.tracker = DailyLossTracker(starting_balance=10_000)
    state.tracker.record_pnl(-50)
    state.save()

    loaded = BotState.load(path)
    assert loaded.has_open_trade()
    assert loaded.last_candle_ts == 42
    assert loaded.tracker is not None
    assert loaded.tracker.today_pnl == -50


def test_load_missing_file_returns_empty_state(tmp_path):
    path = tmp_path / "no-such-state.json"
    loaded = BotState.load(path)
    assert loaded.path == path
    assert loaded.open_trade is None
    assert not loaded.has_open_trade()


def test_save_is_atomic(tmp_path):
    """No leftover tempfiles should be visible in the destination dir after save."""
    path = tmp_path / "state.json"
    state = BotState(path=path, last_candle_ts=1)
    state.save()
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".state-")]
    assert leftovers == []
    # The destination file must contain valid JSON.
    json.loads(path.read_text())


def test_reset_clears_open_trade_and_file(tmp_path):
    path = tmp_path / "state.json"
    state = BotState(path=path, open_trade=_open_trade(), last_candle_ts=42)
    state.save()
    assert path.exists()
    state.reset()
    assert not path.exists()
    assert state.open_trade is None
    assert state.last_candle_ts is None


def test_save_stamps_current_state_version(tmp_path):
    path = tmp_path / "state.json"
    BotState(path=path, last_candle_ts=1).save()
    data = json.loads(path.read_text())
    assert data["state_version"] == STATE_VERSION


def test_load_rejects_unknown_state_version(tmp_path):
    """A state file from a future (or corrupted) schema must surface loudly."""
    path = tmp_path / "state.json"
    path.write_text(json.dumps({
        "state_version": STATE_VERSION + 1,
        "open_trade": None,
        "last_candle_ts": None,
        "tracker": None,
        "extras": {},
    }))
    with pytest.raises(StateVersionMismatch):
        BotState.load(path)


def test_load_accepts_legacy_state_without_version(tmp_path):
    """Files written before versioning landed must still load (treated as v1)."""
    path = tmp_path / "state.json"
    path.write_text(json.dumps({
        "open_trade": None,
        "last_candle_ts": 99,
        "tracker": None,
        "extras": {},
    }))
    loaded = BotState.load(path)
    assert loaded.last_candle_ts == 99


def test_save_creates_lock_file_alongside_state(tmp_path):
    """The advisory lock file is created so multi-process writes serialise."""
    path = tmp_path / "state.json"
    BotState(path=path, last_candle_ts=1).save()
    lock_path = path.with_suffix(".lock")
    assert lock_path.exists(), "expected sibling .lock file"


def test_load_quarantines_corrupt_file_and_returns_empty(tmp_path):
    """A truncated or otherwise unparseable state file must not crash the bot."""
    path = tmp_path / "state.json"
    # Write a malformed JSON payload — e.g. the runner crashed mid-write before
    # the atomic rename, or a disk error left a half-flushed file.
    path.write_text('{"open_trade": null, "last_canDLE_ts')

    loaded = BotState.load(path)
    # Fresh empty state — bot can keep running.
    assert loaded.path == path
    assert loaded.open_trade is None
    assert loaded.last_candle_ts is None

    # Original path is gone (renamed away); a quarantine sibling exists for forensics.
    assert not path.exists()
    quarantined = list(tmp_path.glob("state.corrupt-*"))
    assert len(quarantined) == 1, f"expected 1 quarantine file, got {quarantined}"
    assert "last_canDLE_ts" in quarantined[0].read_text()
