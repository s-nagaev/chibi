"""Tests for the stdio runner CLI entrypoint and tool registration behavior."""

from collections.abc import Iterator
from unittest.mock import patch

import pytest
from click.testing import CliRunner

# Import config first to avoid a circular import when constants is loaded directly.
import chibi.config  # noqa: F401
from chibi.cli import main
from chibi.config import application_settings
from chibi.constants import IDE_STORAGE_ID


@pytest.fixture
def runner() -> CliRunner:
    """Provide a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def restore_client_setting() -> Iterator[None]:
    """Restore the original client value after the test."""
    original = application_settings.client
    yield
    application_settings.client = original


class TestStdioRunnerCLI:
    """CLI dispatch tests for the `chibi stdio` entrypoint."""

    def test_stdio_without_flag_is_rejected(self, runner: CliRunner) -> None:
        """Running `chibi stdio` without a client flag fails with usage guidance."""
        result = runner.invoke(main, ["stdio"])

        assert result.exit_code != 0
        assert "No client selected" in result.output

    def test_stdio_tui_dispatches_and_sets_client(
        self, runner: CliRunner, restore_client_setting: Iterator[None]
    ) -> None:
        """`chibi stdio --tui` sets application_settings.client to "tui" and invokes the runner."""
        with patch("chibi.runners.stdio.run_stdio") as mock_run_stdio:
            result = runner.invoke(main, ["stdio", "--tui"])

        assert result.exit_code == 0
        mock_run_stdio.assert_called_once()
        assert application_settings.client == "tui"

    @pytest.mark.parametrize(
        ("flag", "client"),
        [("--vscode", "vscode"), ("--pycharm", "pycharm"), ("--neovim", "neovim")],
    )
    def test_stdio_client_flag_dispatches_and_sets_client(
        self, runner: CliRunner, restore_client_setting: Iterator[None], flag: str, client: str
    ) -> None:
        """`chibi stdio <client flag>` sets application_settings.client and invokes the runner."""
        with patch("chibi.runners.stdio.run_stdio") as mock_run_stdio:
            result = runner.invoke(main, ["stdio", flag])

        assert result.exit_code == 0
        mock_run_stdio.assert_called_once()
        assert application_settings.client == client

    def test_stdio_rejects_multiple_client_flags(
        self, runner: CliRunner, restore_client_setting: Iterator[None]
    ) -> None:
        """`chibi stdio` rejects passing more than one client flag."""
        result = runner.invoke(main, ["stdio", "--tui", "--vscode"])

        assert result.exit_code != 0
        assert "Multiple clients selected" in result.output

    def test_stdio_tui_does_not_accept_arguments(self, runner: CliRunner) -> None:
        """`chibi stdio --tui` does not take positional arguments."""
        result = runner.invoke(main, ["stdio", "--tui", "extra"])

        assert result.exit_code != 0

    def test_ide_command_removed(self, runner: CliRunner) -> None:
        """The removed `chibi ide` command no longer exists."""
        result = runner.invoke(main, ["ide", "--stdio"])

        assert result.exit_code != 0
        assert "No such command" in result.output


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
