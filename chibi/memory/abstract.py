from abc import ABC, abstractmethod
from typing import Optional

from chibi.models import Message


class LongConversationMemory(ABC):
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
    async def search(self, user_id: int, query: str, n_results: int) -> list[dict]:
        """Search archived messages by semantic similarity.

        Args:
            user_id: The user ID.
            query: Search query string.
            n_results: Maximum number of results to return.

        Returns:
            List of search results with metadata.
        """
        pass

    @abstractmethod
    async def delete_old(self, retention_days: int) -> None:
        """Delete archived messages older than retention_days.

        Args:
            retention_days: Number of days to retain messages.
        """
        pass