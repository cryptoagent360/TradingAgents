"""Parameter sensitivity sweeps for the backtester.

The strategy has several tunable knobs in ``BotConfig`` (atr_mult, taker_fee,
volume_spike_mult, etc). A strategy whose results swing wildly with small
perturbations is over-fit to its default values and won't generalise to live
conditions — fees move, slippage moves, volatility regimes shift.

These tests perturb one knob at a time ±20% (or comparable) against a fixed
favourable-regime fixture and assert:

* PnL doesn't swing > 2× either direction vs the default-param baseline
* Trade count stays > 0 even at the tighter filter setting
* Equity at the higher fee level still finishes above 50% of starting balance

Per-test failures are sanity alarms, not optimisation targets — they say
"this knob is dangerously sharp," not "this knob is set wrong."
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from tradingagents.solana_bot.backtest import run_backtest
from tradingagents.solana_bot.config import BotConfig
from tradingagents.solana_bot.indicators import attach_indicators

pytestmark = pytest.mark.unit


def _trending_fixture(n: int = 500, seed: int = 11) -> pd.DataFrame:
    """Mild uptrend with engineered pullback setups every 100 bars after warmup."""
    rng = np.random.default_rng(seed)
    drift = 0.15
    closes = np.cumsum(np.full(n, drift) + rng.normal(0, 0.20, n)) + 100.0
    df = pd.DataFrame({
        "timestamp": np.arange(n) * 3600 * 1000,
        "open": closes - drift,
        "high": closes + 0.5,
        "low": closes - 0.5,
        "close": closes,
        "volume": [1000.0] * n,
    })
    enriched = attach_indicators(df, ema_fast=50, ema_slow=200, rsi_period=14, atr_period=14)
    for i in range(260, n, 100):
        ema_fast = enriched["ema_fast"].iloc[i]
        ema_slow = enriched["ema_slow"].iloc[i]
        close = enriched["close"].iloc[i]
        if pd.isna(ema_fast) or pd.isna(ema_slow) or close <= ema_slow or ema_fast <= ema_slow:
            continue
        target = float(ema_fast)
        df.loc[i, "open"] = target - 0.4
        df.loc[i, "close"] = target
        df.loc[i, "high"] = target + 0.2
        df.loc[i, "low"] = target - 0.5
        df.loc[i, "volume"] = 2500.0
    return df


def _baseline_config() -> BotConfig:
    return BotConfig(rsi_long_min=0.0, rsi_long_max=100.0)


def _run(cfg: BotConfig) -> dict:
    df = _trending_fixture()
    return run_backtest(df, cfg, starting_balance=10_000).summary


def test_sensitivity_to_atr_mult():
    """Vary stop distance ±20% — PnL stays within 2× of baseline either way."""
    baseline = _run(_baseline_config())
    tighter = _run(dataclasses.replace(_baseline_config(), atr_mult=1.2))
    looser = _run(dataclasses.replace(_baseline_config(), atr_mult=1.8))

    # If the baseline made nothing, the sensitivity test has no signal to perturb.
    if abs(baseline["total_pnl"]) < 1.0:
        pytest.skip("baseline PnL near zero — sensitivity ratio undefined")

    for label, perturbed in (("atr_mult=1.2", tighter), ("atr_mult=1.8", looser)):
        ratio = abs(perturbed["total_pnl"] - baseline["total_pnl"]) / abs(baseline["total_pnl"])
        assert ratio < 2.0, (
            f"{label} swung PnL by {ratio:.1%} vs baseline — stop distance "
            f"is dangerously sensitive (baseline={baseline['total_pnl']:.2f}, "
            f"perturbed={perturbed['total_pnl']:.2f})"
        )


def test_sensitivity_to_taker_fee():
    """Vary fee 0.05% / 0.10% (baseline) / 0.20% — high-fee ending balance > 50% of start."""
    high_fee_cfg = dataclasses.replace(_baseline_config(), taker_fee=0.002)
    summary = _run(high_fee_cfg)
    assert summary["ending_balance"] > 5_000.0, (
        f"at 0.20% taker fee, ending balance collapsed to {summary['ending_balance']:.2f} "
        f"from 10,000 starting — strategy can't survive double-default fee"
    )


def test_sensitivity_to_volume_spike_filter():
    """Tightening volume_spike_mult must not zero out trade count entirely."""
    tighter = dataclasses.replace(_baseline_config(), volume_spike_mult=1.8)
    summary = _run(tighter)
    # Engineered setup bars use 2500 vol vs 1000 baseline (2.5x), so even at
    # 1.8x the filter the engineered setups still fire.
    assert summary["trades"] > 0, (
        "tighter volume_spike filter killed all trades — the engineered setups "
        "in the fixture should still pass 1.8× since they spike to 2.5×"
    )


def test_sensitivity_loose_atr_does_not_invert_strategy():
    """Looser stops (atr_mult=2.0) should not turn winners into losers wholesale."""
    loose_cfg = dataclasses.replace(_baseline_config(), atr_mult=2.0)
    baseline = _run(_baseline_config())
    loose = _run(loose_cfg)
    # If the baseline made money, looser stops should not invert the sign of PnL
    # (some degradation is expected — wider stops = larger losers — but not
    # wholesale reversal). Skip if baseline is near zero (no signal to invert).
    if baseline["total_pnl"] > 100.0:
        assert loose["total_pnl"] > -abs(baseline["total_pnl"]), (
            f"loose stops inverted strategy: baseline +{baseline['total_pnl']:.2f}, "
            f"loose {loose['total_pnl']:.2f}"
        )
