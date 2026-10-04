"""Tests for the fire-and-forget archival exception chain in chroma memory.

Regression context: ``_get_or_create_collection`` raises ``ChromaCollectionError``
(a chibi ``MemoryException``), which is NOT a chromadb ``ChromaError``. Several
callers invoked it OUTSIDE their ``try`` blocks, so archival/search failures
escaped the intended handling. These tests pin the fixed behavior.

chromadb 1.5.9 additionally raises a plain builtin ``ValueError`` for
EF-conflict validation ("new: fastembed vs persisted: default") from
``get_or_create_collection`` — the choke points must wrap it into
``ChromaCollectionError`` so it cannot escape the archival chain raw.
"""

from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from chromadb.errors import ChromaError
from typing_extensions import override

from chibi.exceptions import (
    ChromaArchiveError,
    ChromaCollectionError,
    ChromaSearchError,
    MemoryException,
)
from chibi.memory.chroma import (
    ExternalChromaLongConversationMemory,
    InternalChromaLongConversationMemory,
)
from chibi.models import Message

EF_CONFLICT_MSG = (
    "An embedding function already exists in the collection configuration "
    "with a different configuration. conflict: new: fastembed vs persisted: default"
)


class _TestChromaError(ChromaError):
    """Concrete chromadb ``ChromaError`` for tests (``name`` is abstract upstream)."""

    @classmethod
    @override
    def name(cls) -> str:
        return "ChromaError"


@pytest.fixture
def internal_memory():
    """InternalChromaLongConversationMemory without a real PersistentClient."""
    with patch("chromadb.PersistentClient"):
        memory = InternalChromaLongConversationMemory(embedding_function=MagicMock())
    return memory


@pytest.fixture
def external_memory():
    """ExternalChromaLongConversationMemory with a mocked async client."""
    memory = ExternalChromaLongConversationMemory(embedding_function=MagicMock())
    mock_client = MagicMock()
    memory._client = mock_client
    return memory


@pytest.fixture
def message():
    return Message(role="user", content="Hello")


class TestInternalExceptionChain:
    @pytest.mark.asyncio
    async def test_get_last_batch_id_returns_none_on_collection_error(self, internal_memory):
        internal_memory._get_or_create_collection = AsyncMock(side_effect=ChromaCollectionError("ef conflict"))

        result = await internal_memory._get_last_batch_id(user_id=1, thread_id=2)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_last_batch_id_returns_none_on_chroma_error(self, internal_memory):
        """Pre-existing behavior must be preserved: ChromaError still yields None."""
        collection = MagicMock()
        collection.get.side_effect = _TestChromaError("boom")
        internal_memory._get_or_create_collection = AsyncMock(return_value=collection)

        result = await internal_memory._get_last_batch_id(user_id=1, thread_id=2)

        assert result is None

    @pytest.mark.asyncio
    async def test_archive_message_wraps_collection_error(self, internal_memory, message):
        internal_memory._get_or_create_collection = AsyncMock(side_effect=ChromaCollectionError("ef conflict"))

        with pytest.raises(ChromaArchiveError):
            await internal_memory._archive_message(
                msg=message, batch_id="b1", msg_pos=0, prev_batch_id=None, user_id=1, thread_id=2
            )

    @pytest.mark.asyncio
    async def test_archive_message_wraps_chroma_error_from_add(self, internal_memory, message):
        collection = MagicMock()
        collection.add.side_effect = _TestChromaError("add failed")
        internal_memory._get_or_create_collection = AsyncMock(return_value=collection)

        with pytest.raises(ChromaArchiveError):
            await internal_memory._archive_message(
                msg=message, batch_id="b1", msg_pos=0, prev_batch_id=None, user_id=1, thread_id=2
            )

    @pytest.mark.asyncio
    async def test_archive_message_success(self, internal_memory, message):
        collection = MagicMock()
        internal_memory._get_or_create_collection = AsyncMock(return_value=collection)

        await internal_memory._archive_message(
            msg=message, batch_id="b1", msg_pos=0, prev_batch_id=None, user_id=1, thread_id=2
        )

        collection.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_semantic_search_wraps_collection_error(self, internal_memory):
        internal_memory._get_or_create_collection = AsyncMock(side_effect=ChromaCollectionError("ef conflict"))

        with pytest.raises(ChromaSearchError):
            await internal_memory._semantic_search(user_id=1, query="q", n_results=1, thread_id=2)

    @pytest.mark.asyncio
    async def test_get_batch_by_field_returns_empty_on_collection_error(self, internal_memory):
        internal_memory._get_or_create_collection = AsyncMock(side_effect=ChromaCollectionError("ef conflict"))

        result = await internal_memory._get_batch_by_field(user_id=1, batch_id="b1", thread_id=2)

        assert result == []

    @pytest.mark.asyncio
    async def test_get_or_create_collection_wraps_value_error(self, internal_memory):
        """chromadb raises a plain builtin ValueError for EF-conflict validation
        ("new: fastembed vs persisted: default") — it must be wrapped into
        ChromaCollectionError at the choke point, not escape raw."""
        internal_memory._client.get_or_create_collection.side_effect = ValueError(EF_CONFLICT_MSG)

        with pytest.raises(ChromaCollectionError) as exc_info:
            await internal_memory._get_or_create_collection(user_id=1, thread_id=2)

        assert isinstance(exc_info.value, MemoryException)
        assert not isinstance(exc_info.value, ValueError)

    @pytest.mark.asyncio
    async def test_archive_message_wraps_value_error(self, internal_memory, message):
        """Regression: a persisted known/default collection + our EF raises
        ValueError from get_or_create_collection — _archive_message must convert
        it to ChromaArchiveError, not let the raw ValueError escape."""
        internal_memory._client.get_or_create_collection.side_effect = ValueError(EF_CONFLICT_MSG)

        with pytest.raises(ChromaArchiveError):
            await internal_memory._archive_message(
                msg=message, batch_id="b1", msg_pos=0, prev_batch_id=None, user_id=1, thread_id=2
            )

    @pytest.mark.asyncio
    async def test_semantic_search_wraps_value_error(self, internal_memory):
        internal_memory._client.get_or_create_collection.side_effect = ValueError(EF_CONFLICT_MSG)

        with pytest.raises(ChromaSearchError):
            await internal_memory._semantic_search(user_id=1, query="q", n_results=1, thread_id=2)

    @pytest.mark.asyncio
    async def test_get_last_batch_id_returns_none_on_value_error(self, internal_memory):
        internal_memory._client.get_or_create_collection.side_effect = ValueError(EF_CONFLICT_MSG)

        result = await internal_memory._get_last_batch_id(user_id=1, thread_id=2)

        assert result is None


class TestExternalExceptionChain:
    @pytest.mark.asyncio
    async def test_get_last_batch_id_returns_none_on_collection_error(self, external_memory):
        external_memory._get_or_create_collection = AsyncMock(side_effect=ChromaCollectionError("ef conflict"))

        result = await external_memory._get_last_batch_id(user_id=1, thread_id=2)

        assert result is None

    @pytest.mark.asyncio
    async def test_archive_message_wraps_collection_error(self, external_memory, message):
        external_memory._get_or_create_collection = AsyncMock(side_effect=ChromaCollectionError("ef conflict"))

        with pytest.raises(ChromaArchiveError):
            await external_memory._archive_message(
                msg=message, batch_id="b1", msg_pos=0, prev_batch_id=None, user_id=1, thread_id=2
            )

    @pytest.mark.asyncio
    async def test_semantic_search_wraps_collection_error(self, external_memory):
        external_memory._get_or_create_collection = AsyncMock(side_effect=ChromaCollectionError("ef conflict"))

        with pytest.raises(ChromaSearchError):
            await external_memory._semantic_search(user_id=1, query="q", n_results=1, thread_id=2)

    @pytest.mark.asyncio
    async def test_get_batch_by_id_returns_empty_on_collection_error(self, external_memory):
        external_memory._get_or_create_collection = AsyncMock(side_effect=ChromaCollectionError("ef conflict"))

        result = await external_memory._get_batch_by_id(user_id=1, batch_id="b1", thread_id=2)

        assert result == []

    @pytest.mark.asyncio
    async def test_get_next_batch_returns_empty_on_collection_error(self, external_memory):
        external_memory._get_or_create_collection = AsyncMock(side_effect=ChromaCollectionError("ef conflict"))

        result = await external_memory._get_next_batch(user_id=1, current_batch_id="b1", thread_id=2)

        assert result == []

    @pytest.mark.asyncio
    async def test_get_or_create_collection_wraps_value_error(self, external_memory):
        """chromadb raises a plain builtin ValueError for EF-conflict validation —
        it must be wrapped into ChromaCollectionError at the choke point."""
        external_memory._get_client = AsyncMock()
        external_memory._get_client.return_value.get_or_create_collection = AsyncMock(
            side_effect=ValueError(EF_CONFLICT_MSG)
        )

        with pytest.raises(ChromaCollectionError):
            await external_memory._get_or_create_collection(user_id=1, thread_id=2)

    @pytest.mark.asyncio
    async def test_archive_message_wraps_value_error(self, external_memory, message):
        external_memory._get_client = AsyncMock()
        external_memory._get_client.return_value.get_or_create_collection = AsyncMock(
            side_effect=ValueError(EF_CONFLICT_MSG)
        )

        with pytest.raises(ChromaArchiveError):
            await external_memory._archive_message(
                msg=message, batch_id="b1", msg_pos=0, prev_batch_id=None, user_id=1, thread_id=2
            )


class TestBackgroundArchivalFailureLogging:
    @pytest.mark.asyncio
    async def test_task_manager_logs_background_failure_with_traceback(self):
        """A ChromaCollectionError escaping a background archival task must be
        logged (with traceback) by the task manager's done callback."""
        from chibi.services.task_manager import task_manager

        # task_manager is a process-wide singleton; other tests may have
        # triggered shutdown() which blocks run_task.
        task_manager._shutting_down = False

        async def boom() -> None:
            raise ChromaCollectionError("ef conflict")

        with patch("chibi.services.task_manager.logger") as mock_logger:
            task = task_manager.run_task(boom(), user_id=42)
            assert task is not None
            # Give the event loop a chance to run the done callback
            for _ in range(20):
                await asyncio_sleep_tick()
            mock_logger.opt.assert_called_once_with(exception=ANY)
            assert "failed" in mock_logger.opt.return_value.error.call_args[0][0]

    @pytest.mark.asyncio
    async def test_retention_cleanup_swallows_memory_exception(self):
        """perform_retention_cleanup must not crash on MemoryException."""
        from chibi.services.jobs import archive as archive_job

        mock_memory = MagicMock()
        mock_memory.delete_old = AsyncMock(side_effect=MemoryException("cleanup blew up"))

        with patch.object(archive_job, "memory", mock_memory):
            await archive_job.perform_retention_cleanup()

        mock_memory.delete_old.assert_awaited_once()


async def asyncio_sleep_tick() -> None:
    import asyncio

    await asyncio.sleep(0)
