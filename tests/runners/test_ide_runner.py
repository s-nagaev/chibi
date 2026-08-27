"""Tests for the IDE runner CLI entrypoint and tool registration behavior."""

from unittest.mock import patch

import pytest
from click.testing import CliRunner

# Import config first to avoid a circular import when constants is loaded directly.
import chibi.config  # noqa: F401
from chibi.cli import main
from chibi.constants import IDE_STORAGE_ID


@pytest.fixture
def runner() -> CliRunner:
    """Provide a Click CLI test runner."""
    return CliRunner()


class TestIDERunnerCLI:
    """CLI dispatch tests for the `chibi ide` entrypoint."""

    def test_ide_group_shows_help_without_subcommand(self, runner: CliRunner) -> None:
        """Running `chibi ide` without flags or subcommands prints help."""
        result = runner.invoke(main, ["ide"])

        assert result.exit_code == 0
        assert "Run the Chibi IDE interface" in result.output
        assert "--stdio" in result.output

    def test_ide_stdio_flag_dispatches_to_runner(self, runner: CliRunner) -> None:
        """`chibi ide --stdio` invokes the deferred IDE runner entrypoint."""
        with patch("chibi.runners.ide.run_ide") as mock_run_ide:
            result = runner.invoke(main, ["ide", "--stdio"])

        assert result.exit_code == 0
        mock_run_ide.assert_called_once()

    def test_ide_stdio_flag_does_not_accept_arguments(self, runner: CliRunner) -> None:
        """`chibi ide --stdio` does not take positional arguments."""
        result = runner.invoke(main, ["ide", "--stdio", "extra"])

        assert result.exit_code != 0


class TestIDEStorageIdentity:
    """Tests for the IDE storage identity constant."""

    def test_ide_storage_id_is_negative_large_int(self) -> None:
        """IDE_STORAGE_ID is a large negative integer outside Telegram chat-id range."""
        assert IDE_STORAGE_ID == -(10**16)
        assert isinstance(IDE_STORAGE_ID, int)


class TestRenameThreadToolRegistration:
    """Tests for conditional registration of the Telegram-only rename tool."""

    def test_rename_thread_registered_when_telegram_runner_loaded(self) -> None:
        """RenameThreadTool registers when the Telegram runner module is loaded."""
        import chibi.runners.telegram  # noqa: F401
        from chibi.services.providers.tools.topic import RenameThreadTool

        assert RenameThreadTool.register is True

    def test_rename_thread_not_registered_without_telegram_runner(self) -> None:
        """RenameThreadTool does not register when Telegram runner is not loaded.

        This is verified in a subprocess so that the current process's import
        state does not influence the result.
        """
        import subprocess
        import sys

        code = (
            "from chibi.services.providers.tools.topic import RenameThreadTool; "
            "assert RenameThreadTool.register is False; "
            "print('ok')"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "ok"
