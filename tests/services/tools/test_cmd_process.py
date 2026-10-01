"""Tests for platform-aware process handling in run_command_in_terminal."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chibi.services.providers.tools.cmd import RunCommandInTerminalTool
from chibi.services.providers.tools.exceptions import ToolException


class TestRunCommandInTerminalProcessHandling:
    """Tests for shell spawn kwargs and timeout kill dispatch."""

    @pytest.fixture
    def _mock_moderation(self):
        """Return mocked moderation provider and dependencies."""
        moderator = MagicMock()
        moderator.moderate_command = AsyncMock(return_value=MagicMock(verdict="accepted", reason=""))
        moderator.name = "test_moderator"
        with patch(
            "chibi.services.providers.tools.cmd.get_moderation_provider",
            new=AsyncMock(return_value=moderator),
        ):
            yield moderator

    @pytest.mark.asyncio
    async def test_spawn_uses_platform_process_group_kwargs(self, _mock_moderation):
        """The subprocess is spawned with kwargs provided by the platform-aware helper."""
        spawn_mock = AsyncMock(return_value=MagicMock(returncode=0, communicate=AsyncMock(return_value=(b"", b""))))
        with (
            patch(
                "chibi.services.providers.tools.cmd.get_new_process_group_kwargs",
                return_value={"start_new_session": True},
            ) as kwargs_mock,
            patch("asyncio.create_subprocess_shell", new=spawn_mock),
        ):
            await RunCommandInTerminalTool.function(cmd="echo test", user_id=1, cwd="/tmp")

        kwargs_mock.assert_called_once()
        assert spawn_mock.call_args.kwargs["start_new_session"] is True

    @pytest.mark.asyncio
    async def test_timeout_kills_via_shared_helper(self, _mock_moderation):
        """A timed-out command is terminated through the shared kill helper."""
        mock_process = MagicMock()
        mock_process.pid = 42

        async def slow_communicate() -> tuple[bytes, bytes]:
            await asyncio.sleep(5)
            return b"", b""

        mock_process.communicate = slow_communicate
        with (
            patch(
                "chibi.services.providers.tools.cmd.get_new_process_group_kwargs",
                return_value={"start_new_session": True},
            ),
            patch("asyncio.create_subprocess_shell", new=AsyncMock(return_value=mock_process)),
            patch("chibi.services.providers.tools.cmd.kill_process_tree", new=AsyncMock()) as kill_mock,
        ):
            with pytest.raises(ToolException, match="timed out"):
                await RunCommandInTerminalTool.function(cmd="sleep 30", user_id=1, cwd="/tmp", timeout=1)

        kill_mock.assert_awaited_once_with(mock_process)
