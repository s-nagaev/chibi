"""Tests for the stdio runner CLI entrypoint and tool registration behavior."""

import importlib
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from click.testing import CliRunner

# Import config first to avoid a circular import when constants is loaded directly.
import chibi.config  # noqa: F401
from chibi.cli import main
from chibi.config import application_settings
from chibi.constants import IDE_STORAGE_ID
from chibi.services.providers.tools import RegisteredChibiTools
from chibi.services.providers.tools import topic as topic_module


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

    @pytest.mark.usefixtures("restore_client_setting")
    def test_stdio_tui_dispatches_and_sets_client(self, runner: CliRunner) -> None:
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
    @pytest.mark.usefixtures("restore_client_setting")
    def test_stdio_client_flag_dispatches_and_sets_client(self, runner: CliRunner, flag: str, client: str) -> None:
        """`chibi stdio <client flag>` sets application_settings.client and invokes the runner."""
        with patch("chibi.runners.stdio.run_stdio") as mock_run_stdio:
            result = runner.invoke(main, ["stdio", flag])

        assert result.exit_code == 0
        mock_run_stdio.assert_called_once()
        assert application_settings.client == client

    @pytest.mark.usefixtures("restore_client_setting")
    def test_stdio_rejects_multiple_client_flags(self, runner: CliRunner) -> None:
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
    """Tests for the runner-aware registration gate of the rename tool."""

    @pytest.fixture()
    def restore_tool_registry(self) -> Iterator[None]:
        """Snapshot the tool registry and the rename module around a reload."""
        snapshot = dict(RegisteredChibiTools.tools_map)
        yield
        importlib.reload(topic_module)
        RegisteredChibiTools.tools_map.clear()
        RegisteredChibiTools.tools_map.update(snapshot)

    @pytest.mark.usefixtures("restore_tool_registry")
    @pytest.mark.parametrize("client", ["telegram", "tui", "vscode", "pycharm", "neovim"])
    def test_rename_thread_registers_for_supported_clients(self, client: str) -> None:
        """RenameThreadTool registers for the Telegram bot and every stdio client."""
        with patch("chibi.config.application_settings", SimpleNamespace(client=client)):
            reloaded = importlib.reload(topic_module)

            assert reloaded.RenameThreadTool.register is True
            assert RegisteredChibiTools.tools_map["rename_thread"] is reloaded.RenameThreadTool

    @pytest.mark.usefixtures("restore_tool_registry")
    def test_rename_thread_gate_excludes_terminal_client(self) -> None:
        """The terminal REPL (client='terminal') never registers the rename tool."""
        RegisteredChibiTools.tools_map.pop("rename_thread", None)

        with patch("chibi.config.application_settings", SimpleNamespace(client="terminal")):
            reloaded = importlib.reload(topic_module)

            assert reloaded.RenameThreadTool.register is False
            assert "rename_thread" not in RegisteredChibiTools.tools_map

    @pytest.mark.asyncio
    async def test_terminal_entrypoint_deregisters_rename_tool(self) -> None:
        """End-to-end: the real terminal entrypoint must not expose the rename tool.

        Regression guard: importing ``chibi.runners.terminal`` transitively
        imports the tools module while ``application_settings.client`` still
        holds the ``"telegram"`` default, so the tool registers at import
        time — ``run_chibi`` must deregister it before the REPL starts.
        """
        import chibi.runners.terminal as terminal_module
        from chibi.services.providers.tools.topic import RenameThreadTool

        original_client = application_settings.client
        snapshot = dict(RegisteredChibiTools.tools_map)
        RegisteredChibiTools.register(RenameThreadTool)
        assert "rename_thread" in RegisteredChibiTools.tools_map

        class _StubRunner:
            """Stand-in for TerminalRunner so run_chibi never starts the REPL."""

            def __init__(self, user_id: int) -> None:
                self.user_id = user_id

            async def run(self) -> None:
                return None

        try:
            with (
                patch.object(terminal_module, "TerminalRunner", _StubRunner),
                patch.object(terminal_module, "setup_logging"),
            ):
                await terminal_module.run_chibi(user_id=1)

            assert application_settings.client == "terminal"
            assert "rename_thread" not in RegisteredChibiTools.tools_map
        finally:
            application_settings.client = original_client
            RegisteredChibiTools.tools_map.clear()
            RegisteredChibiTools.tools_map.update(snapshot)
