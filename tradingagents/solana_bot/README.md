# Solana trend + pullback bot

Rule-based long-only trading bot for SOL/USDT (or any ccxt-supported pair).
Lives in `tradingagents.solana_bot` as a sibling to the LLM agent stack —
no LLM in the hot loop.

## Strategy

A trade fires only when **all five** conditions are true on the most
recently closed bar:

1. Price above EMA200 — macro trend is up
2. EMA50 above EMA200 — trend is confirmed
3. Price within `pullback_pct` (default 2%) of EMA50 — price has pulled back to value
4. RSI(14) inside `[40, 55]` — momentum cooled but not bearish
5. Bullish candle with body ≥ 50% of range, on volume ≥ 1.5× the prior 20-bar average

On entry the bot risks **1% of account** per trade, with a stop at
`entry − 1.5 × ATR(14)`. The position scales out:

- **TP1 = +1R**: close 50%, pull stop to breakeven
- **TP2 = +2R**: close 25%
- **Runner**: trail the final 25% with a chandelier-style ATR trail

A daily loss limit of **3%** trips a kill switch that stops all new
trades for the rest of the UTC day. Maximum one open trade at a time.

## Status

- ✅ Backtester (`tradingagents solana backtest`)
- ✅ Paper-trade runner (`tradingagents solana paper`)
- ⏳ **Live execution is intentionally not wired in this release.** Run
  the backtester on at least 3–6 months of data, then paper-trade for
  1–2 weeks before requesting the live PR.

## Quick start

```bash
# Backtest 16 months of SOL on the 1H timeframe
tradingagents solana backtest \
    --symbol SOL/USDT --timeframe 1h \
    --start 2025-01-01 --end 2026-05-01 \
    --balance 10000

# Paper trade against live Binance OHLCV (simulated fills, no API keys needed)
tradingagents solana paper --symbol SOL/USDT --timeframe 1h --balance 10000
```

Outputs land in `~/.tradingagents/solana_bot/`:

- `cache/` — cached OHLCV CSVs (one per pair × timeframe)
- `backtests/<UTC timestamp>/` — `trades.csv`, `equity.csv`, `summary.json`
- `state.json` — paper-runner persisted state (open trade + daily PnL)

## Configuration

Defaults live in `BotConfig` (see `config.py`). Tunable knobs:
`ema_fast`, `ema_slow`, `rsi_period`, `rsi_long_min`, `rsi_long_max`,
`pullback_pct`, `volume_lookback`, `volume_spike_mult`, `atr_period`,
`atr_mult`, `risk_pct`, `max_daily_loss_pct`, `tp1_r`, `tp2_r`,
`tp1_close_fraction`, `tp2_close_fraction`, `trail_atr_mult`,
`taker_fee`.

## Module map

| File             | Purpose                                                      |
| ---------------- | ------------------------------------------------------------ |
| `config.py`      | `BotConfig` dataclass + path helpers                         |
| `indicators.py`  | EMA, RSI, ATR, bullish-candle / volume-spike checks          |
| `signals.py`     | `check_long_setup` — five-condition gate                     |
| `risk.py`        | Position sizing + daily-loss kill switch                     |
| `trade.py`       | `OpenTrade` lifecycle: TP1 → BE stop → TP2 → ATR trail       |
| `execution.py`   | `PaperEngine` (simulated fills) + stubbed `LiveEngine`       |
| `data.py`        | ccxt OHLCV fetcher with on-disk CSV cache                    |
| `state.py`       | Atomic JSON persistence so a kill -9 doesn't lose state      |
| `backtest.py`    | Event-driven backtester replaying the same code paths        |
| `runner.py`      | Paper-trade main loop                                        |

## Pre-live checklist (for the follow-up PR)

Before live execution lands, the operator must:

1. Backtest on at least 3–6 months of recent SOL/USDT data.
2. Confirm win rate ≥ 35%, avg R ≥ 0.4, max drawdown ≤ 12%.
3. Run paper for 1–2 weeks against a moving market.
4. Generate Binance API keys with **trade only — no withdraw permission**.
5. Start on testnet for at least 48 hours; verify orders fill,
   stops trigger, partial closes settle, and state survives a forced restart.
