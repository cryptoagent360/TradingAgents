"""Append-only trade-journal unit tests."""

from __future__ import annotations

import json

import pytest

from tradingagents.solana_bot.journal import TradeJournal
from tradingagents.solana_bot.trade import FillEvent, OpenTrade

pytestmark = pytest.mark.unit


def _open_trade() -> OpenTrade:
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


def _read_lines(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_record_open_writes_a_well_formed_line(tmp_path):
    j = TradeJournal(tmp_path / "journal.jsonl")
    j.record_open(
        trade_id="abc", symbol="SOL/USDT", trade=_open_trade(),
        cycle_number=42, bar_timestamp=1_700_000_000_000,
    )
    [entry] = _read_lines(tmp_path / "journal.jsonl")
    assert entry["event"] == "open"
    assert entry["trade_id"] == "abc"
    assert entry["symbol"] == "SOL/USDT"
    assert entry["entry"] == 100.0
    assert entry["initial_stop"] == 97.0
    assert entry["tp1_price"] == 103.0
    assert entry["tp2_price"] == 106.0
    assert entry["cycle_number"] == 42
    assert entry["bar_timestamp"] == 1_700_000_000_000
    assert "ts" in entry


def test_clock_is_injectable(tmp_path):
    """Tests can pin the clock without monkeypatching the journal module."""
    j = TradeJournal(tmp_path / "journal.jsonl", clock=lambda: "2026-05-07T12:00:00+00:00")
    j.record_open(trade_id="abc", symbol="SOL/USDT", trade=_open_trade())
    [entry] = _read_lines(tmp_path / "journal.jsonl")
    assert entry["ts"] == "2026-05-07T12:00:00+00:00"


def test_record_fill_writes_a_well_formed_line(tmp_path):
    j = TradeJournal(tmp_path / "journal.jsonl")
    fill = FillEvent(kind="tp1", price=103.0, size=5.0)
    j.record_fill(trade_id="abc", symbol="SOL/USDT", fill=fill, realised_pnl=15.0)
    [entry] = _read_lines(tmp_path / "journal.jsonl")
    assert entry["event"] == "fill"
    assert entry["kind"] == "tp1"
    assert entry["price"] == 103.0
    assert entry["size"] == 5.0
    assert entry["realised_pnl"] == 15.0


def test_record_close_writes_a_well_formed_line(tmp_path):
    j = TradeJournal(tmp_path / "journal.jsonl")
    j.record_close(trade_id="abc", symbol="SOL/USDT", total_pnl=42.5, r_multiple=1.42)
    [entry] = _read_lines(tmp_path / "journal.jsonl")
    assert entry["event"] == "close"
    assert entry["total_pnl"] == 42.5
    assert entry["r_multiple"] == 1.42


def test_journal_appends_to_existing_file(tmp_path):
    """Multiple events accumulate as separate lines, in order."""
    path = tmp_path / "journal.jsonl"
    j = TradeJournal(path)
    j.record_open(trade_id="abc", symbol="SOL/USDT", trade=_open_trade())
    j.record_fill(
        trade_id="abc",
        symbol="SOL/USDT",
        fill=FillEvent(kind="tp1", price=103.0, size=5.0),
        realised_pnl=15.0,
    )
    j.record_close(trade_id="abc", symbol="SOL/USDT", total_pnl=15.0, r_multiple=1.0)
    lines = _read_lines(path)
    assert [e["event"] for e in lines] == ["open", "fill", "close"]


def test_journal_creates_parent_directory(tmp_path):
    """A nested path with no parent dir present must Just Work."""
    path = tmp_path / "deep" / "nested" / "journal.jsonl"
    j = TradeJournal(path)
    j.record_open(trade_id="abc", symbol="SOL/USDT", trade=_open_trade())
    assert path.exists()
