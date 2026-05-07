"""BotConfig validation tests."""

import math

import pytest

from tradingagents.solana_bot.config import BotConfig, BotConfigError

pytestmark = pytest.mark.unit


def test_default_config_is_valid():
    """The shipped defaults must construct cleanly."""
    BotConfig()


def test_rejects_unknown_timeframe():
    with pytest.raises(BotConfigError, match="timeframe"):
        BotConfig(timeframe="2h")


def test_rejects_malformed_symbol():
    with pytest.raises(BotConfigError, match="symbol"):
        BotConfig(symbol="SOLUSDT")  # missing /


def test_rejects_negative_risk_pct():
    with pytest.raises(BotConfigError, match="risk_pct"):
        BotConfig(risk_pct=-0.01)


def test_rejects_risk_pct_above_one():
    with pytest.raises(BotConfigError, match="risk_pct"):
        BotConfig(risk_pct=1.5)


def test_rejects_nan_atr_mult():
    with pytest.raises(BotConfigError, match="atr_mult"):
        BotConfig(atr_mult=float("nan"))


def test_rejects_zero_ema_periods():
    with pytest.raises(BotConfigError, match="ema_fast"):
        BotConfig(ema_fast=0)


def test_rejects_ema_fast_not_below_ema_slow():
    with pytest.raises(BotConfigError, match="ema_fast"):
        BotConfig(ema_fast=200, ema_slow=100)


def test_rejects_tp2_not_above_tp1():
    with pytest.raises(BotConfigError, match="tp2_r"):
        BotConfig(tp1_r=2.0, tp2_r=1.0)


def test_rejects_close_fractions_summing_to_one_or_more():
    """The runner is the remainder of size — fractions must leave room for it."""
    with pytest.raises(BotConfigError, match="tp1_close_fraction"):
        BotConfig(tp1_close_fraction=0.6, tp2_close_fraction=0.5)


def test_rejects_rsi_band_inverted():
    with pytest.raises(BotConfigError, match="rsi_long_min"):
        BotConfig(rsi_long_min=70.0, rsi_long_max=30.0)


def test_rejects_rsi_outside_zero_to_hundred():
    with pytest.raises(BotConfigError, match="rsi_long_max"):
        BotConfig(rsi_long_max=150.0)


def test_rejects_negative_max_live_balance():
    with pytest.raises(BotConfigError, match="max_live_balance"):
        BotConfig(max_live_balance=-10.0)
