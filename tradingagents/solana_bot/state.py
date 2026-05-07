"""Crash-safe persisted state for the runner.

A single JSON file under ``~/.tradingagents/solana_bot/state.json`` (path
overridable via ``BotConfig``) holds everything the runner needs to
resume after a kill -9: the open trade (if any), the daily-loss tracker,
and the timestamp of the last processed bar.

Writes are atomic — a tempfile in the same directory is written, then
renamed over the destination — so a crash mid-write cannot leave the
state corrupted.

Each save acquires an exclusive file lock (``fcntl`` advisory lock on a
sibling ``.lock`` file) to serialise multi-process writes. Two bot
processes pointed at the same state path won't shred each other's
on-disk file. Lock semantics are advisory: the lock protects against
file corruption, not against logical interleaving — operators should
still avoid running two instances against the same state path.

Each save stamps a ``schema_version`` field. Loading a file with a
different version raises ``StateSchemaMismatch`` so corrupted or stale
state is surfaced loudly instead of being silently misinterpreted.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from tradingagents.solana_bot.risk import DailyLossTracker
from tradingagents.solana_bot.trade import OpenTrade

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

try:
    import fcntl

    _HAS_FCNTL = True
except ImportError:  # Windows
    _HAS_FCNTL = False


class StateSchemaMismatch(Exception):
    """Raised when a state file's schema_version does not match the runtime."""


@contextmanager
def _file_lock(lock_path: Path) -> Iterator[None]:
    """Acquire an exclusive advisory lock on ``lock_path``. No-op on Windows."""
    if not _HAS_FCNTL:
        logger.warning(
            "fcntl unavailable on this platform; state file lock is a no-op. "
            "Do not run multiple bot instances against the same state path."
        )
        yield
        return
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)


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
            "schema_version": SCHEMA_VERSION,
            "open_trade": self.open_trade.to_dict() if self.open_trade else None,
            "last_candle_ts": self.last_candle_ts,
            "tracker": self.tracker.to_dict() if self.tracker else None,
            "extras": self.extras,
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2)
        lock_path = self.path.with_suffix(".lock")
        with _file_lock(lock_path):
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
        # Missing schema_version is treated as the current version (backward
        # compat with state files written before versioning landed). Any
        # mismatch — including a future version we don't know about — is
        # surfaced loudly.
        version = data.get("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise StateSchemaMismatch(
                f"state file at {path} has schema_version={version}, "
                f"runtime expects {SCHEMA_VERSION}. Manual migration required: "
                f"either delete the file (loses open trade + daily PnL) or "
                f"hand-edit the schema_version after verifying the format."
            )
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
        # Best-effort cleanup of the lock file too. A concurrent reader
        # holding it would prevent unlink; ignore in that case.
        lock_path = self.path.with_suffix(".lock")
        try:
            lock_path.unlink()
        except OSError:
            pass
