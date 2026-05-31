from abc import ABC, abstractmethod

import ulid
from chromadb import GetResult
from pydantic import BaseModel

from chibi.config import application_settings
from chibi.models import Message
from chibi.utils.app import SingletonMeta

EDGE_THRESHOLD = 2


class MemorySearchResult(BaseModel):
    content: str
    role: str
    timestamp: str
    message_id: str
    batch_id: str | None
    msg_pos: int | None
    prev_batch_id: str | None
    thread_id: int | None


class ArchiveState(BaseModel):
    batch_id: str
    prev_batch_id: str | None = None
    next_msg_pos: int
    token_count: int


class LongConversationMemory(ABC, metaclass=SingletonMeta):
    """Abstract base class for long-term conversation memory storage."""

    @staticmethod
    def _get_batch_token_limit() -> int:
        """Get configured batch token limit."""
        return application_settings.batch_token_limit

    @staticmethod
    def _generate_batch_id() -> str:
        """Generate chronologically ordered unique batch ID via ULID."""
        return str(ulid.ulid())

    @staticmethod
    def _format_batch_results(result: GetResult) -> list[MemorySearchResult]:
        """Format raw ChromaDB collection.get() result into search results.

        Args:
            result: Raw result from ChromaDB (dict with 'documents', 'metadatas').

        Returns:
            List of formatted MemorySearchResult dicts; empty list if no data.
        """

        formatted: list[MemorySearchResult] = []
        documents = result.get("documents")
        metadatas = result.get("metadatas")

        if documents and metadatas:
            for doc, metadata in zip(documents, metadatas):
                formatted.append(
                    MemorySearchResult(
                        content=doc,
                        role=str(metadata.get("role", "")),
                        timestamp=str(metadata.get("timestamp", "")),
                        message_id=str(metadata.get("message_id", "")),
                        batch_id=str(metadata.get("batch_id", "")),
                        msg_pos=int(str(metadata.get("msg_pos", -1))),
                        prev_batch_id=str(metadata.get("prev_batch_id", "")) or None,
                        thread_id=int(str(metadata.get("thread_id", 0))),
                    )
                )

        return formatted

    @abstractmethod
    async def archive(self, user_id: int, messages: list[Message], thread_id: int = 0) -> None:
        """Archive messages to vector storage.

        Args:
            user_id: The user ID.
            messages: List of messages to archive.
            thread_id: Thread ID (default: 0).
        """
        pass

    @abstractmethod
    async def search(self, user_id: int, query: str, n_results: int, thread_id: int = 0) -> list[MemorySearchResult]:
        """Search archived messages by semantic similarity.

        Args:
            user_id: The user ID.
            query: Search query string.
            n_results: Maximum number of results to return.
            thread_id: Thread ID (default: 0).

        Returns:
            List of search results.
        """
        pass

    @abstractmethod
    async def delete_old(self, retention_days: int) -> None:
        """Delete archived messages older than retention_days.

        Args:
            retention_days: Number of days to retain messages.
        """
        pass

    def _get_collection_name(self, user_id: int, thread_id: int = 0) -> str:
        """Get collection name for user.

        Args:
            user_id: The user ID.
            thread_id: Thread ID (default: 0).

        Returns:
            Collection name string.
        """
        if thread_id:
            return f"user_{user_id}_thread_{thread_id}"
        return f"user_{user_id}"
