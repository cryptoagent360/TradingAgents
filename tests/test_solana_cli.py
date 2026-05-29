"""CLI integration smoke tests for the solana subcommand."""

import pytest
from typer.testing import CliRunner

from cli.main import app

pytestmark = pytest.mark.unit


runner = CliRunner()


def test_solana_help_shows_subcommands():
    result = runner.invoke(app, ["solana", "--help"])
    assert result.exit_code == 0
    assert "backtest" in result.stdout
    assert "paper" in result.stdout


def test_solana_live_subcommand_refuses_without_explicit_acks():
    """Live invocation without --i-have-verified-trade-only-key OR with
    EXECUTE_TRADES=False refuses cleanly. The shipped default has
    EXECUTE_TRADES=False so this is the path that fires here."""
    result = runner.invoke(app, ["solana", "live"])
    assert result.exit_code == 2
    # Refuses for either of the two valid reasons depending on shipped defaults.
    assert (
        "EXECUTE_TRADES" in result.stdout
        or "trade-only-key" in result.stdout
        or "Refusing to start" in result.stdout
    )
