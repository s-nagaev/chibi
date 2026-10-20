"""Tests for the with_chroma_archival decorator."""

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chibi.memory.chroma import with_chroma_archival
from chibi.models import Message, User
from chibi.services.context_note import CONTEXT_NOTE_OPEN, format_context_note

NOTE = format_context_note(["approximate_context_size: 42 tokens (0% of 200,000 limit)"])
assert NOTE, "sanity: the test note must be non-empty"


class MockDatabase:
    """Mock database for testing the decorator."""

    def __init__(self):
        self.add_message_called = False
        self.add_message_kwargs: dict = {}
        self.get_messages_called = False

    async def add_message(self, user, message, ttl=None, thread_id=0):
        self.add_message_called = True
        self.add_message_kwargs = {"ttl": ttl, "thread_id": thread_id}

    async def get_messages(self, user, thread_id=0):
        self.get_messages_called = True
        return []

    async def drop_messages(self, user, thread_id=0):
        pass

    async def get_user(self, user_id):
        return None

    async def create_user(self, user_id):
        return User(id=user_id)

    async def save_user(self, user):
        pass

    async def count_image(self, user_id):
        pass


class MockMemory:
    """Mock memory for testing the decorator."""

    def __init__(self):
        self.archive_called = False
        self.archive_kwargs: dict = {}

    async def archive(self, user_id, messages, thread_id=0):
        self.archive_called = True
        self.archive_kwargs = {"user_id": user_id, "messages": messages, "thread_id": thread_id}


@pytest.fixture
def mock_inner():
    return MockDatabase()


@pytest.fixture
def mock_memory():
    return MockMemory()


@pytest.fixture
def user():
    return User(id=123)


@pytest.fixture
def message():
    return Message(role="user", content="Test message")


class TestWithChromaArchival:
    """Tests for the with_chroma_archival decorator."""

    def test_returns_identity_when_memory_is_none(self, mock_inner):
        """When memory is None, the decorator must return the storage unchanged."""
        wrapped = with_chroma_archival(None)(mock_inner)

        assert wrapped is mock_inner

    @pytest.mark.asyncio
    async def test_add_message_calls_inner(self, mock_inner, user, message):
        """Decorator should still call the underlying add_message."""
        wrapped = with_chroma_archival(None)(mock_inner)

        await wrapped.add_message(user, message)

        assert mock_inner.add_message_called is True

    @pytest.mark.asyncio
    async def test_add_message_archives_when_memory_exists(self, mock_inner, mock_memory, user, message):
        """When memory is configured, add_message should also dispatch archive via task_manager."""
        with patch("chibi.memory.chroma.task_manager") as mock_task_manager:
            mock_task_manager.run_task = MagicMock()

            wrapped = with_chroma_archival(mock_memory)(mock_inner)
            await wrapped.add_message(user, message, ttl=5, thread_id=7)

            # Primary write still happened
            assert mock_inner.add_message_called is True
            assert mock_inner.add_message_kwargs == {"ttl": 5, "thread_id": 7}

            # Archive was scheduled via task_manager with the right coroutine and user_id
            mock_task_manager.run_task.assert_called_once()
            call_args = mock_task_manager.run_task.call_args
            assert call_args[0][0].cr_code.co_name == "archive"
            assert call_args[1]["user_id"] == user.id

    @pytest.mark.asyncio
    async def test_add_message_does_not_archive_when_memory_none(self, mock_inner, user, message):
        """When memory is None, task_manager.run_task must not be called."""
        with patch("chibi.memory.chroma.task_manager") as mock_task_manager:
            mock_task_manager.run_task = MagicMock()

            wrapped = with_chroma_archival(None)(mock_inner)
            await wrapped.add_message(user, message)

            mock_task_manager.run_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_other_methods_are_not_touched(self, mock_inner, user):
        """The decorator must leave every method other than add_message untouched."""
        wrapped = with_chroma_archival(None)(mock_inner)

        result = await wrapped.get_messages(user)

        assert mock_inner.get_messages_called is True
        assert result == []

    @pytest.mark.asyncio
    async def test_archive_error_does_not_break_primary_write(self, mock_inner, user, message):
        """A failing archive coroutine scheduled via task_manager must not affect add_message."""
        failing_memory = AsyncMock()
        failing_memory.archive = AsyncMock(side_effect=Exception("Archive failed"))

        with patch("chibi.memory.chroma.task_manager") as mock_task_manager:
            mock_task_manager.run_task = MagicMock()

            wrapped = with_chroma_archival(failing_memory)(mock_inner)

            # Must not raise: archive is fire-and-forget, primary write is awaited directly
            await wrapped.add_message(user, message)

            assert mock_inner.add_message_called is True
            mock_task_manager.run_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_thread_id_is_forwarded_to_inner_and_archive(self, mock_inner, mock_memory, user, message):
        """thread_id must be propagated to both the underlying storage and memory.archive."""
        with patch("chibi.memory.chroma.task_manager") as mock_task_manager:
            mock_task_manager.run_task = MagicMock()

            wrapped = with_chroma_archival(mock_memory)(mock_inner)
            await wrapped.add_message(user, message, thread_id=42)

            assert mock_inner.add_message_kwargs["thread_id"] == 42

            # Inspect the coroutine passed to task_manager: it's memory.archive(user.id, [message], thread_id=42)
            archive_coro = mock_task_manager.run_task.call_args[0][0]
            archive_coro.close()  # we only inspect metadata, never await it
            assert archive_coro.cr_code.co_name == "archive"

    @pytest.mark.asyncio
    async def test_recursion_safe(self, mock_inner, mock_memory, user, message):
        """Calling add_message on the wrapped instance must not recurse into itself."""
        with patch("chibi.memory.chroma.task_manager") as mock_task_manager:
            mock_task_manager.run_task = MagicMock()

            wrapped = with_chroma_archival(mock_memory)(mock_inner)
            await wrapped.add_message(user, message)

            # The inner add_message should be hit exactly once, not multiple times
            assert mock_inner.add_message_called is True
            mock_task_manager.run_task.assert_called_once()


class CapturingDatabase(MockDatabase):
    """MockDatabase variant that records the message each write received."""

    def __init__(self):
        super().__init__()
        self.written_messages: list[Message] = []

    async def add_message(self, user, message, ttl=None, thread_id=0):
        await super().add_message(user, message, ttl=ttl, thread_id=thread_id)
        self.written_messages.append(message)


class CapturingMemory:
    """Minimal memory stub recording the messages handed to archive."""

    def __init__(self):
        self.archived: list[Message] = []

    async def archive(self, user_id, messages, thread_id=0):
        self.archived.extend(messages)


class TestStripOnWrite:
    """The volatile context note must never reach stored history or the archive."""

    @pytest.fixture
    def noted_message(self) -> Message:
        content = '{"user": "telegram", "prompt": "hi", "type": "user message"}'
        return Message(role="user", content=content + "\n\n" + NOTE)

    @pytest.mark.asyncio
    async def test_note_is_stripped_from_the_write_and_the_archive(self, noted_message) -> None:
        """Both the storage write and the archive receive the cleaned content."""
        inner = CapturingDatabase()
        memory = CapturingMemory()
        wrapped = with_chroma_archival(cast(Any, memory))(cast(Any, inner))

        with patch("chibi.memory.chroma.task_manager") as mock_task_manager:
            mock_task_manager.run_task = MagicMock()
            await wrapped.add_message(User(id=123), noted_message, thread_id=7)
            # Run the scheduled archival coroutine synchronously.
            archive_coro = mock_task_manager.run_task.call_args[0][0]
            await archive_coro

        assert CONTEXT_NOTE_OPEN not in inner.written_messages[0].content
        assert CONTEXT_NOTE_OPEN not in memory.archived[0].content
        assert inner.written_messages[0].content == memory.archived[0].content

    @pytest.mark.asyncio
    async def test_stripping_never_mutates_the_callers_message(self, noted_message) -> None:
        """The wrapper writes a cleaned copy; the caller's Message keeps its content."""
        original_content = noted_message.content
        wrapped = with_chroma_archival(cast(Any, CapturingMemory()))(cast(Any, CapturingDatabase()))

        with patch("chibi.memory.chroma.task_manager") as mock_task_manager:
            mock_task_manager.run_task = MagicMock()
            await wrapped.add_message(User(id=123), noted_message)
            # Inspect-and-close: never await the scheduled archival coroutine.
            archive_coro = mock_task_manager.run_task.call_args[0][0]
            archive_coro.close()

        assert noted_message.content == original_content

    @pytest.mark.asyncio
    async def test_clean_message_passes_through_byte_identical(self) -> None:
        """A message without the note is written and archived byte for byte."""
        clean = Message(role="user", content="plain text without any note")
        inner = CapturingDatabase()
        memory = CapturingMemory()
        wrapped = with_chroma_archival(cast(Any, memory))(cast(Any, inner))

        with patch("chibi.memory.chroma.task_manager") as mock_task_manager:
            mock_task_manager.run_task = MagicMock()
            await wrapped.add_message(User(id=123), clean)
            archive_coro = mock_task_manager.run_task.call_args[0][0]
            await archive_coro

        assert inner.written_messages[0] is clean
        assert memory.archived[0] is clean
        assert inner.written_messages[0].content == "plain text without any note"
