"""Tests for run_command_in_terminal decoding behavior."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chibi.services.providers.tools.cmd import RunCommandInTerminalTool


class TestRunCommandInTerminalDecode:
    """Tests for subprocess output decoding in RunCommandInTerminalTool."""

    @pytest.fixture
    def mock_moderation(self):
        """Return mocked moderation provider and dependencies."""
        moderator = MagicMock()
        moderator.moderate_command = AsyncMock(return_value=MagicMock(verdict="accepted", reason=""))
        moderator.name = "test_moderator"
        with patch(
            "chibi.services.providers.tools.cmd.get_moderation_provider",
            new=AsyncMock(return_value=moderator),
        ):
            yield moderator

    @pytest.fixture
    def mock_process(self):
        """Return a mocked subprocess process."""
        process = MagicMock()
        process.pid = 42
        process.returncode = 0
        process.communicate = AsyncMock()
        process.kill = MagicMock()
        process.wait = AsyncMock()
        return process

    @pytest.mark.asyncio
    async def test_decodes_invalid_utf8_with_replacement(self, mock_moderation, mock_process):
        """Invalid UTF-8 bytes in stdout/stderr must be replaced, not raise."""
        # 0xff is never valid UTF-8
        mock_process.communicate.return_value = (b"valid \xff text", b"err \xfe data")

        with patch("asyncio.create_subprocess_shell", new=AsyncMock(return_value=mock_process)):
            result = await RunCommandInTerminalTool.function(
                cmd="echo test",
                user_id=1,
                cwd="/tmp",
            )

        assert result["return_code"] == 0
        assert "valid \ufffd text" in result["stdout"]
        assert "err \ufffd data" in result["stderr"]
