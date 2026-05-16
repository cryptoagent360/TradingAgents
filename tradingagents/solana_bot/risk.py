"""Position sizing and daily-loss circuit breaker.

The risk-per-trade is fixed at ``config.risk_pct`` of starting balance.
Position size derives from the distance between entry and stop, not
from a target leverage — this keeps risk constant across volatility
regimes.

A trading day is a UTC calendar day. Realised PnL accumulates into
``today_pnl``; once cumulative loss exceeds ``max_daily_loss_pct``, the
kill switch trips and stays tripped until the next UTC day (the runner
must restart to clear it).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


def position_size(balance: float, entry: float, stop: float, risk_pct: float) -> float:
    """Risk-based position size in base-asset units.

    Raises ``ValueError`` if entry ≤ stop (no risk distance) or balance
    ≤ 0 — both indicate caller bugs that we must not paper over.
    """
    if balance <= 0:
        raise ValueError(f"balance must be positive, got {balance}")
    if entry <= stop:
        raise ValueError(f"entry {entry} must be above stop {stop} for a long")
    if not 0 < risk_pct < 1:
        raise ValueError(f"risk_pct must be in (0, 1), got {risk_pct}")
    risk_amount = balance * risk_pct
    distance = entry - stop
    return risk_amount / distance


@dataclass
class DailyLossTracker:
    """Accumulates realised PnL for the current UTC day.

    Exposed via the runner's persisted state so a process restart inside
    the same UTC day preserves the kill-switch decision.
    """

    starting_balance: float
    max_daily_loss_pct: float = 0.03
    today_pnl: float = 0.0
    today_date_utc: str = ""
    kill_switch_triggered: bool = False

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).date().isoformat()

    def _maybe_rollover(self) -> None:
        today = self._today()
        if self.today_date_utc != today:
            self.today_date_utc = today
            self.today_pnl = 0.0
            self.kill_switch_triggered = False

    def record_pnl(self, pnl: float) -> None:
        self._maybe_rollover()
        self.today_pnl += pnl
        if self.today_pnl <= -self.max_daily_loss_pct * self.starting_balance:
            self.kill_switch_triggered = True

    def can_open_trade(self) -> bool:
        self._maybe_rollover()
        return not self.kill_switch_triggered

    def to_dict(self) -> dict:
        return {
            "starting_balance": self.starting_balance,
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "today_pnl": self.today_pnl,
            "today_date_utc": self.today_date_utc,
            "kill_switch_triggered": self.kill_switch_triggered,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DailyLossTracker":
        return cls(**data)
