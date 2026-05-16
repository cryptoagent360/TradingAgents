"""OHLCV fetcher (Binance spot via ccxt) with on-disk CSV cache.

Public market data — no API keys required. Caches per
(symbol, timeframe) into ``~/.tradingagents/solana_bot/cache/``. Cache
files store all history fetched so far; subsequent calls only request
the bars after the last cached timestamp.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from tradingagents.solana_bot.config import BotConfig

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
TIMEFRAME_TO_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


def _safe_filename(symbol: str, timeframe: str) -> str:
    return f"{symbol.replace('/', '_')}-{timeframe}.csv"


def _build_exchange(exchange_id: str):
    import ccxt

    if not hasattr(ccxt, exchange_id):
        raise ValueError(f"unknown ccxt exchange: {exchange_id}")
    klass = getattr(ccxt, exchange_id)
    return klass({"enableRateLimit": True})


def _to_ms(date_str: str) -> int:
    dt = pd.to_datetime(date_str, utc=True).to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_ohlcv_range(
    config: BotConfig,
    *,
    start: str,
    end: Optional[str] = None,
    exchange=None,
) -> pd.DataFrame:
    """Fetch OHLCV bars in ``[start, end]`` with on-disk caching.

    ``start`` and ``end`` accept any pandas-parseable date string. ``end``
    defaults to "now". Returns a DataFrame indexed 0..N-1 with columns
    ``timestamp`` (ms epoch, UTC), ``open``, ``high``, ``low``, ``close``,
    ``volume``.
    """
    cache_dir = config.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / _safe_filename(config.symbol, config.timeframe)

    if cache_file.exists():
        cached = pd.read_csv(cache_file)
    else:
        cached = pd.DataFrame(columns=OHLCV_COLUMNS)

    start_ms = _to_ms(start)
    end_ms = _to_ms(end) if end else int(time.time() * 1000)

    last_cached_ms = int(cached["timestamp"].max()) if not cached.empty else 0
    fetch_since_ms = max(start_ms, last_cached_ms + 1)

    if fetch_since_ms <= end_ms:
        new_rows = _download(config, since_ms=fetch_since_ms, until_ms=end_ms, exchange=exchange)
        if not new_rows.empty:
            cached = pd.concat([cached, new_rows], ignore_index=True)
            cached = cached.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            cached.to_csv(cache_file, index=False)

    mask = (cached["timestamp"] >= start_ms) & (cached["timestamp"] <= end_ms)
    return cached.loc[mask].reset_index(drop=True)


def _download(config: BotConfig, *, since_ms: int, until_ms: int, exchange=None) -> pd.DataFrame:
    ex = exchange if exchange is not None else _build_exchange(config.exchange)
    timeframe_seconds = TIMEFRAME_TO_SECONDS.get(config.timeframe)
    if timeframe_seconds is None:
        raise ValueError(f"unsupported timeframe: {config.timeframe}")

    all_rows: list[list[float]] = []
    cursor = since_ms
    limit = 1000
    while cursor <= until_ms:
        batch = ex.fetch_ohlcv(config.symbol, timeframe=config.timeframe, since=cursor, limit=limit)
        if not batch:
            break
        all_rows.extend(batch)
        last_ms = batch[-1][0]
        next_cursor = last_ms + timeframe_seconds * 1000
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(batch) < limit:
            break

    if not all_rows:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    df = pd.DataFrame(all_rows, columns=OHLCV_COLUMNS)
    df = df[df["timestamp"] <= until_ms]
    return df


def fetch_latest_closed_bars(config: BotConfig, lookback_bars: int, *, exchange=None) -> pd.DataFrame:
    """Fetch the last ``lookback_bars`` closed bars for the live runner."""
    seconds = TIMEFRAME_TO_SECONDS[config.timeframe]
    now_ms = int(time.time() * 1000)
    since_ms = now_ms - (lookback_bars + 2) * seconds * 1000
    last_closed_ms = (now_ms // (seconds * 1000)) * seconds * 1000 - 1
    df = fetch_ohlcv_range(
        config,
        start=datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc).isoformat(),
        end=datetime.fromtimestamp(last_closed_ms / 1000, tz=timezone.utc).isoformat(),
        exchange=exchange,
    )
    return df


def sleep_until_next_candle(timeframe: str, *, sleeper=time.sleep, now=time.time) -> None:
    """Block until the next candle close + a small buffer for exchange lag."""
    seconds = TIMEFRAME_TO_SECONDS[timeframe]
    now_s = now()
    next_close_s = (int(now_s) // seconds + 1) * seconds
    sleeper(max(0.0, next_close_s - now_s) + 2.0)
