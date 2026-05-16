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


def test_solana_live_subcommand_refuses():
    result = runner.invoke(app, ["solana", "live"])
    assert result.exit_code == 2
    assert "not yet wired" in result.stdout or "Live execution" in result.stdout
