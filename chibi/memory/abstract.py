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


class LongConversationMemory(ABC, metaclass=SingletonMeta):
    """Abstract base class for long-term conversation memory storage."""

    @abstractmethod
    async def archive(self, user_id: int, messages: list[Message]) -> None:
        """Archive messages to vector storage.

        Args:
            user_id: The user ID.
            messages: List of messages to archive.
        """
        pass

    @abstractmethod
    async def search(self, user_id: int, query: str, n_results: int) -> list[MemorySearchResult]:
        """Search archived messages by semantic similarity.

        Args:
            user_id: The user ID.
            query: Search query string.
            n_results: Maximum number of results to return.

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

    def _get_collection_name(self, user_id: int) -> str:
        """Get collection name for user.

        Args:
            user_id: The user ID.

        Returns:
            Collection name string.
        """
        return f"user_{user_id}"
