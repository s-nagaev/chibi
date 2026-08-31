"""Tests for IDE transport thread cloning (/new_thread_with_current_context) and rename_thread."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

import chibi.config  # noqa: F401
from chibi.constants import IDE_STORAGE_ID
from chibi.exceptions import StorageError
from chibi.models import Message
from chibi.runners.ide_transport import COMMANDS, IDEInterface, IDEStdioRunner
from chibi.storage.local import LocalStorage

SOURCE_THREAD = 7
DESTINATION_THREAD = 42


class OutputRecorder:
    """Capture emitted JSONL protocol frames."""

    def __init__(self) -> None:
        """Initialize an empty frame list."""
        self.frames: list[dict[str, Any]] = []

    async def __call__(self, frame: dict[str, Any]) -> None:
        """Store one emitted frame.

        Args:
            frame: Protocol frame to capture.
        """
        self.frames.append(frame)


def request(request_id: str, prompt: str = "hello", thread_id: int = DESTINATION_THREAD) -> dict[str, Any]:
    """Build a valid IDE request frame.

    Args:
        request_id: Request correlation identifier.
        prompt: User prompt or slash command.
        thread_id: The thread the request travels on (the clone destination).

    Returns:
        A valid request frame.
    """
    return {
        "type": "request",
        "request_id": request_id,
        "thread_id": thread_id,
        "prompt": prompt,
        "workspace_root": "/tmp",
        "active_file": None,
        "selection": None,
        "cursor_position": None,
        "language_id": None,
    }


def local_db_provider(db: LocalStorage) -> SimpleNamespace:
    """Build a fake database provider serving one LocalStorage instance.

    Args:
        db: The storage instance the injected database calls resolve to.

    Returns:
        Provider namespace compatible with ``chibi.storage.database._db_provider``.
    """
    return SimpleNamespace(get_database=AsyncMock(return_value=db))


async def seed_thread_history(db: LocalStorage, thread_id: int, count: int) -> None:
    """Persist ``count`` user messages into one thread of the local storage.

    Args:
        db: Storage instance to seed.
        thread_id: Thread bucket to write into.
        count: Number of messages to append.
    """
    user = await db.get_or_create_user(user_id=IDE_STORAGE_ID)
    for index in range(count):
        await db.add_message(user=user, message=Message(role="user", content=f"message-{index}"), thread_id=thread_id)


async def run_to_completion(runner: IDEStdioRunner, recorder: OutputRecorder, frame: dict[str, Any]) -> None:
    """Dispatch initialize plus one request frame and wait for its terminal frame.

    Args:
        runner: Transport under test with ``_write`` patched to the recorder.
        recorder: Frame capture sink.
        frame: The request frame to dispatch.
    """
    await runner._handle_message({"type": "initialize", "protocol_version": 1})
    await runner._handle_message(frame)
    for _ in range(200):
        if any(
            frame_.get("type") in ("result", "error") and frame_.get("request_id") == frame["request_id"]
            for frame_ in recorder.frames
        ):
            return
        await asyncio.sleep(0.01)
    pytest.fail(f"Request {frame['request_id']} did not produce a terminal frame.")


def terminal_frame(recorder: OutputRecorder, request_id: str) -> dict[str, Any]:
    """Return the terminal (result or error) frame emitted for one request.

    Args:
        recorder: Frame capture sink.
        request_id: Request correlation identifier.

    Returns:
        The terminal protocol frame (status frames are skipped).
    """
    return next(
        frame
        for frame in recorder.frames
        if frame.get("request_id") == request_id and frame.get("type") in ("result", "error")
    )


class TestCloneCommandAdvertisement:
    """Capabilities and help advertising for the clone command."""

    @pytest.mark.asyncio
    async def test_capabilities_advertise_new_thread_with_current_context(self) -> None:
        """The clone command is advertised in capabilities.commands after handshake."""
        recorder = OutputRecorder()
        runner = IDEStdioRunner()
        with patch.object(runner, "_write", recorder):
            await runner._handle_message({"type": "initialize", "protocol_version": 1})

        ready = next(frame for frame in recorder.frames if frame.get("type") == "ready")
        assert "/new_thread_with_current_context" in ready["capabilities"]["commands"]
        assert "/new_thread_with_current_context" in COMMANDS

    @pytest.mark.asyncio
    async def test_help_lists_new_thread_with_current_context(self) -> None:
        """/help output includes the clone command."""
        recorder = OutputRecorder()
        runner = IDEStdioRunner()
        with patch.object(runner, "_write", recorder):
            await run_to_completion(runner, recorder, request("help-clone", "/help"))

        result = terminal_frame(recorder, "help-clone")
        assert result["type"] == "result"
        assert "/new_thread_with_current_context" in result["content"]


class TestCloneHappyPath:
    """Successful clones via the mocked pipeline (handler-level contract).

    Note: the db-level re-keying itself is owned by
    ``chibi.services.user.clone_thread_messages``, which is asserted here only
    as a call contract (old -> new ids, name); its internals are out of scope
    for the IDE transport (and its core must remain byte-identical).
    """

    @pytest.mark.asyncio
    async def test_clone_rekeys_old_to_new_registers_name_and_acks(self) -> None:
        """A named clone dispatches the core clone old->new with the given name and acks."""
        source_history = [Message(role="user", content=f"message-{index}") for index in range(3)]

        async def fake_thread_messages(storage_id: int, thread_id: int) -> list[Message]:
            if storage_id == IDE_STORAGE_ID and thread_id == SOURCE_THREAD:
                return list(source_history)
            return []

        recorder = OutputRecorder()
        runner = IDEStdioRunner()
        clone_mock = AsyncMock(return_value=3)
        with (
            patch.object(runner, "_write", recorder),
            patch("chibi.runners.ide_transport._get_thread_messages", new=fake_thread_messages),
            patch("chibi.runners.ide_transport.clone_thread_messages", clone_mock),
        ):
            await run_to_completion(
                runner, recorder, request("clone-ok", f"/new_thread_with_current_context {SOURCE_THREAD} My Clone")
            )

        clone_mock.assert_awaited_once_with(
            storage_id=IDE_STORAGE_ID,
            old_thread_id=SOURCE_THREAD,
            new_thread_id=DESTINATION_THREAD,
            name="My Clone",
        )
        result = terminal_frame(recorder, "clone-ok")
        assert result["type"] == "result"
        assert "Thread cloned" in result["content"]
        assert "My Clone" in result["content"]
        assert f"ID: {DESTINATION_THREAD}" in result["content"]
        assert "3 messages copied" in result["content"]

    @pytest.mark.asyncio
    async def test_clone_without_name_uses_clone_default_naming(self) -> None:
        """An unnamed clone passes name=None so the clone logic applies its default naming."""
        source_history = [Message(role="user", content=f"message-{index}") for index in range(2)]

        async def fake_thread_messages(storage_id: int, thread_id: int) -> list[Message]:
            if storage_id == IDE_STORAGE_ID and thread_id == SOURCE_THREAD:
                return list(source_history)
            return []

        recorder = OutputRecorder()
        runner = IDEStdioRunner()
        clone_mock = AsyncMock(return_value=2)
        with (
            patch.object(runner, "_write", recorder),
            patch("chibi.runners.ide_transport._get_thread_messages", new=fake_thread_messages),
            patch("chibi.runners.ide_transport.clone_thread_messages", clone_mock),
        ):
            await run_to_completion(
                runner, recorder, request("clone-default", f"/new_thread_with_current_context {SOURCE_THREAD}")
            )

        clone_mock.assert_awaited_once_with(
            storage_id=IDE_STORAGE_ID,
            old_thread_id=SOURCE_THREAD,
            new_thread_id=DESTINATION_THREAD,
            name=None,
        )
        result = terminal_frame(recorder, "clone-default")
        assert result["type"] == "result"
        assert f"ID: {DESTINATION_THREAD}" in result["content"]
        assert "2 messages copied" in result["content"]


class TestCloneErrorPaths:
    """Readable rejection of invalid clone requests (mocked pipeline)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("raw_source", ["abc", "-5", "12.5", ""])
    async def test_clone_invalid_source_id_is_rejected(self, raw_source: str) -> None:
        """Non-negative-integer validation rejects malformed source ids before any db access."""
        recorder = OutputRecorder()
        runner = IDEStdioRunner()
        clone_mock = AsyncMock()
        prompt = f"/new_thread_with_current_context {raw_source}".strip()
        with (
            patch.object(runner, "_write", recorder),
            patch("chibi.runners.ide_transport.clone_thread_messages", clone_mock),
        ):
            await run_to_completion(runner, recorder, request("clone-bad-id", prompt))

        error = terminal_frame(recorder, "clone-bad-id")
        assert error["type"] == "error"
        assert error["code"] == "invalid_request"
        assert "Invalid source thread id" in error["message"]
        clone_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_clone_empty_source_history_is_rejected(self, tmp_path: Path) -> None:
        """A source thread without history is refused and nothing is cloned."""
        db = LocalStorage(storage_path=str(tmp_path))
        await seed_thread_history(db, DESTINATION_THREAD, 0)
        recorder = OutputRecorder()
        runner = IDEStdioRunner()
        clone_mock = AsyncMock()
        with (
            patch.object(runner, "_write", recorder),
            patch("chibi.storage.database._db_provider", local_db_provider(db)),
            patch("chibi.runners.ide_transport.clone_thread_messages", clone_mock),
        ):
            await run_to_completion(
                runner, recorder, request("clone-empty", f"/new_thread_with_current_context {SOURCE_THREAD}")
            )

        error = terminal_frame(recorder, "clone-empty")
        assert error["type"] == "error"
        assert error["code"] == "invalid_request"
        assert "no message history" in error["message"]
        assert str(SOURCE_THREAD) in error["message"]
        clone_mock.assert_not_called()
        user = await db.get_or_create_user(user_id=IDE_STORAGE_ID)
        assert DESTINATION_THREAD not in user.thread_names

    @pytest.mark.asyncio
    async def test_clone_into_destination_with_history_is_rejected(self, tmp_path: Path) -> None:
        """A destination bucket that already holds history is protected from mixing."""
        db = LocalStorage(storage_path=str(tmp_path))
        await seed_thread_history(db, SOURCE_THREAD, 2)
        await seed_thread_history(db, DESTINATION_THREAD, 1)
        recorder = OutputRecorder()
        runner = IDEStdioRunner()
        clone_mock = AsyncMock()
        with (
            patch.object(runner, "_write", recorder),
            patch("chibi.storage.database._db_provider", local_db_provider(db)),
            patch("chibi.runners.ide_transport.clone_thread_messages", clone_mock),
        ):
            await run_to_completion(
                runner, recorder, request("clone-collide", f"/new_thread_with_current_context {SOURCE_THREAD} X")
            )

        error = terminal_frame(recorder, "clone-collide")
        assert error["type"] == "error"
        assert error["code"] == "invalid_request"
        assert "already has message history" in error["message"]
        clone_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_clone_storage_failure_returns_readable_error(self, tmp_path: Path) -> None:
        """A StorageError from the clone logic becomes a readable request_failed error."""
        db = LocalStorage(storage_path=str(tmp_path))
        await seed_thread_history(db, SOURCE_THREAD, 2)
        recorder = OutputRecorder()
        runner = IDEStdioRunner()
        with (
            patch.object(runner, "_write", recorder),
            patch("chibi.storage.database._db_provider", local_db_provider(db)),
            patch(
                "chibi.runners.ide_transport.clone_thread_messages",
                AsyncMock(side_effect=StorageError(detail="redis is unreachable")),
            ),
        ):
            await run_to_completion(
                runner, recorder, request("clone-db-fail", f"/new_thread_with_current_context {SOURCE_THREAD} X")
            )

        error = terminal_frame(recorder, "clone-db-fail")
        assert error["type"] == "error"
        assert error["code"] == "request_failed"
        assert f"Failed to clone thread {SOURCE_THREAD}" in error["message"]
        assert "redis is unreachable" in error["message"]

    @pytest.mark.asyncio
    async def test_clone_unexpected_failure_is_sanitized(self, tmp_path: Path) -> None:
        """Unexpected clone failures stay readable and never leak the raw exception text."""
        db = LocalStorage(storage_path=str(tmp_path))
        await seed_thread_history(db, SOURCE_THREAD, 2)
        recorder = OutputRecorder()
        runner = IDEStdioRunner()
        with (
            patch.object(runner, "_write", recorder),
            patch("chibi.storage.database._db_provider", local_db_provider(db)),
            patch(
                "chibi.runners.ide_transport.clone_thread_messages",
                AsyncMock(side_effect=RuntimeError("secret-db-boom-details")),
            ),
        ):
            await run_to_completion(
                runner, recorder, request("clone-crash", f"/new_thread_with_current_context {SOURCE_THREAD} X")
            )

        error = terminal_frame(recorder, "clone-crash")
        assert error["type"] == "error"
        assert error["code"] == "request_failed"
        assert f"Failed to clone thread {SOURCE_THREAD}" in error["message"]
        assert "secret-db-boom-details" not in error["message"]

    @pytest.mark.asyncio
    async def test_clone_refused_while_source_thread_is_busy(self) -> None:
        """A source thread with an in-flight request is refused before touching storage."""
        recorder = OutputRecorder()
        runner = IDEStdioRunner()
        runner._thread_requests[SOURCE_THREAD] = 1
        clone_mock = AsyncMock()
        with (
            patch.object(runner, "_write", recorder),
            patch("chibi.runners.ide_transport.clone_thread_messages", clone_mock),
        ):
            await run_to_completion(
                runner, recorder, request("clone-busy", f"/new_thread_with_current_context {SOURCE_THREAD} X")
            )

        error = terminal_frame(recorder, "clone-busy")
        assert error["type"] == "error"
        assert error["code"] == "invalid_request"
        assert "busy" in error["message"]
        clone_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_concurrent_clone_of_same_source_is_refused(self, tmp_path: Path) -> None:
        """A second clone of the same source is refused while the first is still running."""
        db = LocalStorage(storage_path=str(tmp_path))
        await seed_thread_history(db, SOURCE_THREAD, 3)
        recorder = OutputRecorder()
        runner = IDEStdioRunner()
        release = asyncio.Event()

        async def blocking_clone(*args: Any, **kwargs: Any) -> int:
            await release.wait()
            return 3

        with (
            patch.object(runner, "_write", recorder),
            patch("chibi.storage.database._db_provider", local_db_provider(db)),
            patch("chibi.runners.ide_transport.clone_thread_messages", new=blocking_clone),
        ):
            await runner._handle_message({"type": "initialize", "protocol_version": 1})
            await runner._handle_message(
                request("clone-first", f"/new_thread_with_current_context {SOURCE_THREAD} First", thread_id=10)
            )
            for _ in range(200):
                if SOURCE_THREAD in runner._active_clones:
                    break
                await asyncio.sleep(0.01)
            assert SOURCE_THREAD in runner._active_clones

            await runner._handle_message(
                request("clone-second", f"/new_thread_with_current_context {SOURCE_THREAD} Second", thread_id=11)
            )
            for _ in range(200):
                if any(
                    frame.get("request_id") == "clone-second" and frame.get("type") == "error"
                    for frame in recorder.frames
                ):
                    break
                await asyncio.sleep(0.01)

            release.set()
            for _ in range(200):
                if any(
                    frame.get("request_id") == "clone-first" and frame.get("type") == "result"
                    for frame in recorder.frames
                ):
                    break
                await asyncio.sleep(0.01)

        second = terminal_frame(recorder, "clone-second")
        assert second["type"] == "error"
        assert second["code"] == "invalid_request"
        assert "already in progress" in second["message"]

        first = terminal_frame(recorder, "clone-first")
        assert first["type"] == "result"
        assert "Thread cloned" in first["content"]
        assert SOURCE_THREAD not in runner._active_clones, "Clone guard must be released after completion"


class TestRenameThread:
    """IDEInterface.rename_thread persists names via save_thread_name."""

    @pytest.mark.asyncio
    async def test_rename_thread_delegates_to_save_thread_name(self) -> None:
        """rename_thread calls save_thread_name for the request's thread and returns True."""
        interface = IDEInterface(thread_id=DESTINATION_THREAD, prompt="p", context={}, emit=lambda _: None)
        with patch("chibi.runners.ide_transport.save_thread_name", new=AsyncMock()) as save_mock:
            result = await interface.rename_thread("Renamed Thread")

        assert result is True
        save_mock.assert_awaited_once_with(
            storage_id=IDE_STORAGE_ID, thread_id=DESTINATION_THREAD, name="Renamed Thread"
        )

    @pytest.mark.asyncio
    async def test_rename_thread_persists_to_backend_storage(self, tmp_path: Path) -> None:
        """A rename through the IDE interface lands in the user's thread-name map."""
        db = LocalStorage(storage_path=str(tmp_path))
        interface = IDEInterface(thread_id=DESTINATION_THREAD, prompt="p", context={}, emit=lambda _: None)
        with patch("chibi.storage.database._db_provider", local_db_provider(db)):
            result = await interface.rename_thread("Renamed Thread")

        assert result is True
        user = await db.get_or_create_user(user_id=IDE_STORAGE_ID)
        assert user.thread_names[DESTINATION_THREAD] == "Renamed Thread"
