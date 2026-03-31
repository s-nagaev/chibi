"""Tests for ChromaDecoratedStorage decorator."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chibi.models import Message, User
from chibi.storage.chroma_decorator import ChromaDecoratedStorage


class MockDatabase:
    """Mock database for testing."""

    def __init__(self):
        self.add_message_called = False
        self.get_messages_called = False

    async def add_message(self, user, message, ttl=None):
        self.add_message_called = True

    async def get_messages(self, user):
        self.get_messages_called = True
        return []

    async def drop_messages(self, user):
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
    """Mock memory for testing."""

    def __init__(self):
        self.archive_called = False

    async def archive(self, user_id, messages):
        self.archive_called = True


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


class TestChromaDecoratedStorage:
    """Tests for ChromaDecoratedStorage."""

    @pytest.mark.asyncio
    async def test_add_message_calls_inner(self, mock_inner, user, message):
        """Test that add_message calls inner storage."""
        decorator = ChromaDecoratedStorage(mock_inner, memory=None)

        await decorator.add_message(user, message)

        assert mock_inner.add_message_called is True

    @pytest.mark.asyncio
    async def test_add_message_calls_archive_when_memory_exists(self, mock_inner, mock_memory, user, message):
        """Test that archive is called when memory is configured."""
        # Patch task_manager to avoid singleton issues in tests
        with patch("chibi.storage.chroma_decorator.task_manager") as mock_task_manager:
            mock_task_manager.run_task = MagicMock()

            decorator = ChromaDecoratedStorage(mock_inner, memory=mock_memory)

            await decorator.add_message(user, message)

            # Verify run_task was called with archive coroutine
            mock_task_manager.run_task.assert_called_once()
            call_args = mock_task_manager.run_task.call_args
            assert call_args[0][0].cr_code.co_name == "archive"
            assert call_args[1]["user_id"] == user.id

    @pytest.mark.asyncio
    async def test_add_message_does_not_archive_when_memory_none(self, mock_inner, user, message):
        """Test that archive is NOT called when memory is None."""
        with patch("chibi.storage.chroma_decorator.task_manager") as mock_task_manager:
            mock_task_manager.run_task = MagicMock()

            decorator = ChromaDecoratedStorage(mock_inner, memory=None)

            await decorator.add_message(user, message)

            # task_manager.run_task should NOT be called when memory is None
            mock_task_manager.run_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_messages_proxies_to_inner(self, mock_inner, user):
        """Test that get_messages proxies to inner storage."""
        mock_inner.get_messages = AsyncMock(return_value=[{"role": "user", "content": "test"}])
        decorator = ChromaDecoratedStorage(mock_inner, memory=None)

        result = await decorator.get_messages(user)

        mock_inner.get_messages.assert_called_once_with(user)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_drop_messages_proxies_to_inner(self, mock_inner, user):
        """Test that drop_messages proxies to inner storage."""
        mock_inner.drop_messages = AsyncMock()
        decorator = ChromaDecoratedStorage(mock_inner, memory=None)

        await decorator.drop_messages(user)

        mock_inner.drop_messages.assert_called_once_with(user)

    @pytest.mark.asyncio
    async def test_get_user_proxies_to_inner(self, mock_inner):
        """Test that get_user proxies to inner storage."""
        mock_user = User(id=123)
        mock_inner.get_user = AsyncMock(return_value=mock_user)
        decorator = ChromaDecoratedStorage(mock_inner, memory=None)

        result = await decorator.get_user(123)

        assert result == mock_user
        mock_inner.get_user.assert_called_once_with(123)

    @pytest.mark.asyncio
    async def test_create_user_proxies_to_inner(self, mock_inner):
        """Test that create_user proxies to inner storage."""
        mock_inner.create_user = AsyncMock(return_value=User(id=123))
        decorator = ChromaDecoratedStorage(mock_inner, memory=None)

        result = await decorator.create_user(123)

        assert result.id == 123
        mock_inner.create_user.assert_called_once_with(123)

    @pytest.mark.asyncio
    async def test_archive_error_is_handled(self, mock_inner, user, message):
        """Test that errors in archive don't propagate."""
        failing_memory = AsyncMock()
        failing_memory.archive = AsyncMock(side_effect=Exception("Archive failed"))

        # Patch task_manager to avoid actual background task
        with patch("chibi.storage.chroma_decorator.task_manager") as mock_task_manager:
            mock_task_manager.run_task = MagicMock()

            decorator = ChromaDecoratedStorage(mock_inner, memory=failing_memory)

            # This should NOT raise an exception
            await decorator.add_message(user, message)

            # Verify inner was called (primary storage worked)
            assert mock_inner.add_message_called is True
