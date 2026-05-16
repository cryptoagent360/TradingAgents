"""Best-effort operational alerting via Telegram Bot API.

A blank ``TELEGRAM_BOT_TOKEN`` or ``TELEGRAM_CHAT_ID`` makes the notifier
a no-op — operators who don't want alerts don't need to do anything.
Send failures (network, HTTP non-200, unexpected exceptions) are logged
and swallowed; alerting must never crash the bot.

Stdlib-only (``urllib.request``) so we don't pull in another HTTP
dependency for two POSTs per day.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"
TELEGRAM_TIMEOUT_S = 5.0


class TelegramNotifier:
    """Sends short text messages to a Telegram chat. No-op if not configured.

    Construct with explicit ``token``/``chat_id`` for tests; in production
    leave them ``None`` and the constructor reads from
    ``TELEGRAM_BOT_TOKEN`` / ``TELEGRAM_CHAT_ID`` env vars. Either being
    blank disables the notifier entirely.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
        *,
        opener: Optional[object] = None,
    ):
        self.token = (token if token is not None else os.environ.get("TELEGRAM_BOT_TOKEN", "")).strip()
        self.chat_id = (chat_id if chat_id is not None else os.environ.get("TELEGRAM_CHAT_ID", "")).strip()
        self._enabled = bool(self.token and self.chat_id)
        # Injection point for tests — defaults to the real urlopen.
        self._opener = opener if opener is not None else urllib.request.urlopen
        if not self._enabled:
            logger.info("Telegram notifier disabled (token or chat_id missing)")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def notify(self, level: str, message: str) -> bool:
        """POST ``[LEVEL] message`` to Telegram. Returns True on HTTP 200.

        Never raises — operational alerts must not be in the critical path
        of the trading loop. Failure to deliver an alert is logged at WARNING.
        """
        if not self._enabled:
            return False
        text = f"[{level.upper()}] {message}"
        url = TELEGRAM_API_URL.format(token=self.token)
        body = json.dumps({"chat_id": self.chat_id, "text": text}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener(req, timeout=TELEGRAM_TIMEOUT_S) as resp:
                status = getattr(resp, "status", None) or getattr(resp, "code", None)
                if status != 200:
                    logger.warning("Telegram notify non-200: HTTP %s", status)
                    return False
                return True
        except urllib.error.URLError as exc:
            logger.warning("Telegram notify network error: %s", exc)
            return False
        except Exception as exc:  # noqa: BLE001 — best-effort: never raise from here
            logger.warning("Telegram notify unexpected error: %s", exc)
            return False
