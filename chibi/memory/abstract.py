from abc import ABC, abstractmethod
from typing import TypedDict

from chibi.models import Message
from chibi.utils.app import SingletonMeta

EDGE_THRESHOLD = 2


class MemorySearchResult(TypedDict):
    content: str
    role: str
    timestamp: str
    message_id: str
    batch_id: str | None
    msg_pos: int | None
    prev_batch_id: str | None
    thread_id: int | None


class LongConversationMemory(ABC, metaclass=SingletonMeta):
    """Abstract base class for long-term conversation memory storage."""

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
