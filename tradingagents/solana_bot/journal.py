"""Append-only trade journal.

Every trade-lifecycle event — open, fill, close — gets a line in
``~/.tradingagents/solana_bot/journal.jsonl`` (path overridable via
``BotConfig``). One JSON object per line, suitable for grep + jq:

    grep '"event": "fill"' journal.jsonl | jq '.realised_pnl' | paste -sd+ - | bc

Distinct from ``state.json`` (which is the *current* runner state and
gets overwritten on every save). The journal is the immutable,
append-only ledger — every fill ever taken is here, in order, forever.

Concurrency: POSIX guarantees ``O_APPEND`` writes ≤ PIPE_BUF bytes
(typically 4096B) are atomic, so two writers can't interleave each
other's lines. Each journal entry is well under that limit. No explicit
lock needed; the state.py file lock prevents the multi-bot scenario
from arising in the first place.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from tradingagents.solana_bot.trade import FillEvent, OpenTrade

logger = logging.getLogger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class TradeJournal:
    """Append-only JSONL ledger of trade-lifecycle events."""

    def __init__(self, path: Path, *, clock: Callable[[], str] = _utcnow_iso):
        self.path = path
        self._clock = clock

    def _append(self, payload: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, separators=(",", ":")) + "\n"
        # Open with mode "a" — POSIX guarantees atomic appends below PIPE_BUF.
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line)

    def record_open(
        self,
        *,
        trade_id: str,
        symbol: str,
        trade: OpenTrade,
        ts: Optional[str] = None,  # caller may override; otherwise uses self._clock
    ) -> None:
        self._append({
            "ts": ts or self._clock(),
            "event": "open",
            "trade_id": trade_id,
            "symbol": symbol,
            "entry": trade.entry,
            "initial_stop": trade.initial_stop,
            "tp1_price": trade.tp1_price,
            "tp2_price": trade.tp2_price,
            "size": trade.size,
            "atr_at_entry": trade.atr_at_entry,
        })

    def record_fill(
        self,
        *,
        trade_id: str,
        symbol: str,
        fill: FillEvent,
        realised_pnl: float,
        ts: Optional[str] = None,  # caller may override; otherwise uses self._clock
    ) -> None:
        self._append({
            "ts": ts or self._clock(),
            "event": "fill",
            "trade_id": trade_id,
            "symbol": symbol,
            "kind": fill.kind,
            "price": fill.price,
            "size": fill.size,
            "realised_pnl": realised_pnl,
        })

    def record_close(
        self,
        *,
        trade_id: str,
        symbol: str,
        total_pnl: float,
        r_multiple: Optional[float] = None,
        ts: Optional[str] = None,  # caller may override; otherwise uses self._clock
    ) -> None:
        self._append({
            "ts": ts or self._clock(),
            "event": "close",
            "trade_id": trade_id,
            "symbol": symbol,
            "total_pnl": total_pnl,
            "r_multiple": r_multiple,
        })
