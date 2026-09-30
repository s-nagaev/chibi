"""Tests for platform-aware subprocess helpers in chibi.utils.process."""

import signal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chibi.utils.process import get_new_process_group_kwargs, kill_process_tree


class TestGetNewProcessGroupKwargs:
    """Tests for subprocess kwargs selecting the process-group isolation mechanism."""

    def test_posix_uses_start_new_session(self):
        """POSIX platforms isolate the child with setsid via start_new_session."""
        with patch("chibi.utils.process.sys") as sys_mock:
            sys_mock.platform = "linux"
            assert get_new_process_group_kwargs() == {"start_new_session": True}

    def test_windows_uses_creationflags(self):
        """Windows uses CREATE_NEW_PROCESS_GROUP because setsid is unavailable."""
        with (
            patch("chibi.utils.process.sys") as sys_mock,
            patch("chibi.utils.process.subprocess") as subprocess_mock,
        ):
            sys_mock.platform = "win32"
            subprocess_mock.CREATE_NEW_PROCESS_GROUP = 512
            assert get_new_process_group_kwargs() == {"creationflags": 512}


class TestKillProcessTreePosix:
    """Tests for the POSIX branch of kill_process_tree."""

    @pytest.fixture
    def mock_process(self):
        """Return a mocked subprocess process."""
        process = MagicMock()
        process.pid = 4242
        process.kill = MagicMock()
        process.wait = AsyncMock()
        return process

    async def test_kills_process_group_then_falls_back_to_process_kill(self, mock_process):
        """POSIX sends SIGKILL to the whole process group and then kills the process itself."""
        with (
            patch("chibi.utils.process.sys") as sys_mock,
            patch("os.getpgid", return_value=1234) as getpgid_mock,
            patch("os.killpg") as killpg_mock,
        ):
            sys_mock.platform = "linux"
            await kill_process_tree(mock_process)

        getpgid_mock.assert_called_once_with(4242)
        killpg_mock.assert_called_once_with(1234, signal.SIGKILL)
        mock_process.kill.assert_called_once()
        mock_process.wait.assert_awaited_once()

    async def test_swallows_process_lookup_error_on_killpg(self, mock_process):
        """An already-dead process group does not break the kill sequence."""
        with (
            patch("chibi.utils.process.sys") as sys_mock,
            patch("os.getpgid", return_value=1234),
            patch("os.killpg", side_effect=ProcessLookupError),
        ):
            sys_mock.platform = "linux"
            await kill_process_tree(mock_process)

        mock_process.kill.assert_called_once()
        mock_process.wait.assert_awaited_once()

    async def test_swallows_process_lookup_error_on_kill_and_wait(self, mock_process):
        """A process disappearing between killpg and kill() does not raise."""
        mock_process.kill.side_effect = ProcessLookupError
        mock_process.wait.side_effect = ProcessLookupError

        with (
            patch("chibi.utils.process.sys") as sys_mock,
            patch("os.getpgid", return_value=1234),
            patch("os.killpg"),
        ):
            sys_mock.platform = "linux"
            await kill_process_tree(mock_process)

        mock_process.kill.assert_called_once()
        mock_process.wait.assert_awaited_once()


class TestKillProcessTreeWindows:
    """Tests for the Windows branch of kill_process_tree."""

    @pytest.fixture
    def mock_process(self):
        """Return a mocked subprocess process."""
        process = MagicMock()
        process.pid = 4242
        process.kill = MagicMock()
        process.wait = AsyncMock()
        return process

    async def test_kills_tree_via_taskkill_without_killpg(self, mock_process):
        """Windows force-kills the whole tree via taskkill and never touches POSIX killpg."""
        with (
            patch("chibi.utils.process.sys") as sys_mock,
            patch("chibi.utils.process.subprocess.run") as run_mock,
            patch("os.killpg") as killpg_mock,
        ):
            sys_mock.platform = "win32"
            await kill_process_tree(mock_process)

        run_mock.assert_called_once()
        assert run_mock.call_args.args[0] == ["taskkill", "/F", "/T", "/PID", "4242"]
        killpg_mock.assert_not_called()
        mock_process.kill.assert_called_once()
        mock_process.wait.assert_awaited_once()

    async def test_swallows_taskkill_failure(self, mock_process):
        """A failing taskkill falls back to process.kill() instead of raising."""
        with (
            patch("chibi.utils.process.sys") as sys_mock,
            patch("chibi.utils.process.subprocess.run", side_effect=OSError("taskkill missing")),
            patch("os.killpg") as killpg_mock,
        ):
            sys_mock.platform = "win32"
            await kill_process_tree(mock_process)

        killpg_mock.assert_not_called()
        mock_process.kill.assert_called_once()
        mock_process.wait.assert_awaited_once()
