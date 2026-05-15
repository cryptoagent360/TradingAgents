# CLAUDE.md — context for future Claude sessions in this repo

This file is read on session start. Keep it short, factual, evergreen.

## What this repo is

A LangGraph-based multi-agent trading research stack (`tradingagents/`) plus
a focused Solana trend-pullback bot under `tradingagents/solana_bot/` that
shipped through this session.

## Two related but separate trading bots

The operator runs two bots:

1. **The Solana bot in THIS repo** (`tradingagents/solana_bot/`) — SOL/USDT
   trend-pullback against Binance via ccxt. Paper-mode runs end-to-end; the
   LiveEngine is half-built (constructor + open_long work, `manage()` raises
   NotImplementedError). State persistence, JSONL trade journal, runner-lock,
   AI filter via Claude, Telegram notifier, `TRADING_MODE` preflight, systemd
   unit, deployment guide — all live in `tradingagents/solana_bot/`.

2. **A Kraken bot at `C:\Users\jay\OneDrive\Desktop\Kraken_bot\Kraken_Bot\`**
   on the operator's Windows machine — entry point `kraken_bot_v3.py`. NOT
   in this repo. We have not seen its source. It has its own state, trade
   history, and accumulated tuning the operator has built over weeks.

These two bots do not share state, journals, or config. They are independent
processes.

## Known quirk: kraken_bot_v3.py does NOT call load_dotenv()

`kraken_bot_v3.py` expects env vars to be present in the parent shell BEFORE
launch. Running `python kraken_bot_v3.py` directly will report
`Telegram: disabled` even when `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` are
correctly set in `.env`.

The operator's launcher script `Kraken_Bot\start_bot.ps1` (PowerShell) loads
`.env` into the shell first, then invokes `python kraken_bot_v3.py`. Always
direct the operator to use `.\start_bot.ps1`, not the bare python invocation.

## Solana bot — useful starting points

- `tradingagents/solana_bot/runner.py` — main `run_paper` entry; reconcile +
  state lifecycle + journal calls + Telegram notify on kill switch / drift.
- `tradingagents/solana_bot/execution.py` — `PaperEngine` (working) and
  `LiveEngine` (constructor + `open_long` + `reconcile`; `manage` raises).
- `tradingagents/solana_bot/state.py` — atomic writes, fcntl runner-lock,
  rolling backup snapshots, corruption quarantine, schema versioning.
- `tradingagents/solana_bot/journal.py` — append-only JSONL trade log;
  injectable clock; `cycle_number` / `bar_timestamp` fields on every record.
- `tradingagents/solana_bot/notifications.py` — stdlib-only Telegram
  notifier; no-op when creds blank; never raises into the trading loop.
- `sol_bot/ai_filter.py` — Claude-powered binary APPROVE/REJECT gate;
  `EXECUTE_TRADES = False` is a hardcoded source-edit kill switch.
- `bot.py` (repo root) — unattended entry; loads `.env`; `TRADING_MODE`
  preflight refuses anything but `paper` until LiveEngine is fully wired.

## Tests

`python -m pytest tests/ -m unit -q` should pass ~169 tests under 5s. Slow
disk-touching tests are technically integration but are tagged `unit`
because they're still fast — there's a noted cleanup to re-tag them as
`smoke`.

## Sandbox-environment limitation

This Linux sandbox cannot reach `api.binance.com` due to a self-signed cert
in the chain. Any test that actually fetches OHLCV will fail with
`NetworkError: SSL CERTIFICATE_VERIFY_FAILED`. That's an environment issue,
not a bot bug. On the operator's real VPS the fetch succeeds.

## What to NEVER do without explicit operator approval

- Flip `EXECUTE_TRADES = True` in `sol_bot/ai_filter.py`
- Change `TRADING_MODE` from `paper` to anything else in `.env`
- Add new package dependencies
- Modify either bot's source to "help" without being asked
- Echo API tokens or secrets in chat — never, even when pasted in conversation
