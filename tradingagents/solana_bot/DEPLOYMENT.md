# Solana Bot — VPS Deployment Guide

Deploy the bot as a systemd service on a Linux VPS (Ubuntu 22.04+ or
similar). Two phases: **paper-trading burn-in** (72 hours minimum, no
real money) and then **tiny-capital live** (operator opt-in, with
hard safety caps).

## Prerequisites

- Linux VPS with public network egress to `api.binance.com`
- Python 3.11+, `pip`, `git`
- A non-root user (`bot` or similar) for the service to run as
- An `ANTHROPIC_API_KEY` for the AI filter
- A Binance API key (trade-only — **no withdraw**, **no transfer**) once
  you're ready for live mode; not needed during paper burn-in

## Initial setup

```bash
sudo useradd -m -s /bin/bash bot
sudo -u bot -i

git clone https://github.com/cryptoagent360/TradingAgents.git
cd TradingAgents
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
$EDITOR .env  # ANTHROPIC_API_KEY=... (BINANCE_* later)
```

The bot writes its state, journal, backups, and logs under
`~/.tradingagents/solana_bot/` by default.

## Phase 1: paper-trading burn-in (72 hours)

Run the bot in paper mode for at least 72 hours before considering
live execution. Paper mode uses live OHLCV from Binance but simulated
fills — no real orders, no real money.

```bash
cd ~/TradingAgents
source .venv/bin/activate
BOT_BURN_IN_HOURS=72 nohup python bot.py > output.log 2>&1 &
disown
```

Or via the CLI:

```bash
tradingagents solana paper --burn-in-hours 72
```

After 72h the loop exits cleanly with `state.extras.last_cycle =
"burn_in_complete"`.

### What to check after burn-in

```bash
# Did the bot stay up? (log should end with burn_in_complete)
tail -50 output.log

# Did the daily-loss kill switch ever trip?
jq .tracker.kill_switch_triggered ~/.tradingagents/solana_bot/state.json

# How many trades did the strategy take?
grep -c '"event":"open"' ~/.tradingagents/solana_bot/journal.jsonl

# Distribution of outcomes (R-multiples on closed trades)
jq -r 'select(.event=="close") | .r_multiple' \
    ~/.tradingagents/solana_bot/journal.jsonl
```

If the run was clean — no crashes, kill switch did not trip
prematurely, the journal shows reasonable trade behavior — you can
proceed to live with tiny capital.

## Phase 2: tiny-capital live

**Hard rule: start with the equivalent of $50–$100 of trading capital.**
Larger sums get refused at LiveEngine construction
(`max_live_balance` default is $100).

1. Generate a Binance API key with **trade only** scope. Disable
   withdraw and disable internal transfer. Whitelist the VPS IP.

2. Add credentials to `.env`:

   ```ini
   BINANCE_API_KEY=...
   BINANCE_API_SECRET=...
   ```

3. Edit `sol_bot/ai_filter.py` and flip the safeguard:

   ```python
   EXECUTE_TRADES = True   # WAS False
   ```

   Commit this change to git so the live-trading flip is auditable.

4. (Optional, while LiveEngine order placement is still in progress)
   Confirm the LiveEngine constructor itself accepts your account by
   running a one-shot reconcile script — see the LiveEngine tests for
   the call shape.

5. Start the systemd service (see below) and watch closely for the
   first day.

## systemd unit

Drop this at `/etc/systemd/system/solana-bot.service`. A template
lives at `tradingagents/solana_bot/solana-bot.service` in this repo.

```bash
sudo cp tradingagents/solana_bot/solana-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable solana-bot
sudo systemctl start solana-bot
journalctl -u solana-bot -f   # follow logs
```

Restart-on-failure is enabled with a 30s backoff so transient network
errors auto-recover. The runner-lock prevents two instances from
racing on the same state path.

## Log rotation

Bot output goes to stdout/stderr and (under systemd) is captured by
the journal. To keep `output.log` (when running via nohup) bounded,
add a logrotate rule:

```
# /etc/logrotate.d/solana-bot
/home/bot/TradingAgents/output.log {
    daily
    rotate 14
    compress
    missingok
    notifempty
    copytruncate
}
```

## Health checks

A minimal external watchdog can poll for any of:

- `state.json` missing or older than 2× the timeframe → bot is wedged
- `state.tracker.kill_switch_triggered == true` → daily loss limit hit, bot exited
- A `state.corrupt-*` quarantine file appeared → corruption recovery fired

```bash
#!/bin/bash
STATE=~/.tradingagents/solana_bot/state.json
if [ ! -f "$STATE" ]; then echo "NO STATE FILE"; exit 1; fi
AGE=$(( $(date +%s) - $(stat -c %Y "$STATE") ))
if [ $AGE -gt 7200 ]; then echo "STATE STALE (${AGE}s)"; exit 1; fi
KS=$(jq -r '.tracker.kill_switch_triggered' "$STATE")
if [ "$KS" = "true" ]; then echo "KILL SWITCH TRIPPED"; exit 1; fi
echo "OK"
```

## Recovery procedures

### State file corrupted

The bot quarantines bad state files automatically (see
`state.corrupt-<timestamp>` in `~/.tradingagents/solana_bot/`) and
starts fresh. If a real open position existed, restore from a backup
snapshot:

```bash
cd ~/.tradingagents/solana_bot/
ls backups/  # newest snapshots last
cp backups/state-20260507T...Z.json state.json
sudo systemctl restart solana-bot
```

### Kill switch tripped

The daily-loss kill switch (3% by default) is intentional friction —
review the journal, understand what happened, then manually clear
state and restart:

```bash
rm ~/.tradingagents/solana_bot/state.json   # forfeits the open trade if any
sudo systemctl restart solana-bot
```

### Two bots accidentally started

The runner-lock catches this — the second instance refuses with
`RunnerAlreadyActive`. Stop the duplicate via systemd or pkill.
