"""Crash-safe persisted state for the runner.

A single JSON file under ``~/.tradingagents/solana_bot/state.json`` (path
overridable via ``BotConfig``) holds everything the runner needs to
resume after a kill -9: the open trade (if any), the daily-loss tracker,
and the timestamp of the last processed bar.

Writes are atomic — a tempfile in the same directory is written, then
renamed over the destination — so a crash mid-write cannot leave the
state corrupted.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tradingagents.solana_bot.risk import DailyLossTracker
from tradingagents.solana_bot.trade import OpenTrade


@dataclass
class BotState:
    path: Path
    open_trade: Optional[OpenTrade] = None
    last_candle_ts: Optional[int] = None
    tracker: Optional[DailyLossTracker] = None
    extras: dict = field(default_factory=dict)

    def has_open_trade(self) -> bool:
        return self.open_trade is not None and not self.open_trade.is_closed()

    def to_dict(self) -> dict:
        return {
            "open_trade": self.open_trade.to_dict() if self.open_trade else None,
            "last_candle_ts": self.last_candle_ts,
            "tracker": self.tracker.to_dict() if self.tracker else None,
            "extras": self.extras,
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2)
        fd, tmp_path = tempfile.mkstemp(prefix=".state-", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(tmp_path, self.path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path: Path) -> "BotState":
        if not path.exists():
            return cls(path=path)
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        open_trade = OpenTrade.from_dict(data["open_trade"]) if data.get("open_trade") else None
        tracker = DailyLossTracker.from_dict(data["tracker"]) if data.get("tracker") else None
        return cls(
            path=path,
            open_trade=open_trade,
            last_candle_ts=data.get("last_candle_ts"),
            tracker=tracker,
            extras=data.get("extras", {}),
        )

    def reset(self) -> None:
        self.open_trade = None
        self.last_candle_ts = None
        self.tracker = None
        self.extras = {}
        if self.path.exists():
            self.path.unlink()
