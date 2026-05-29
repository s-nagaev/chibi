"""Batch manager for accumulating messages before ChromaDB storage."""
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import TypedDict

import ulid

from chibi.models import Message
from chibi.config import application_settings


class BatchMetadata(TypedDict):
    """Metadata for each message in ChromaDB."""
    message_id: str
    batch_id: str
    msg_pos: int
    prev_batch_id: str | None
    role: str
    timestamp: str


@dataclass
class Batch:
    """Represents a batch of messages."""
    batch_id: str
    prev_batch_id: str | None
    messages: list[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    # _batch_token_limit: int = 30000  # Token limit per batch (not message count)
    _batch_token_limit: int = 2000  # Token limit per batch (not message count)

    @property
    def estimated_tokens(self) -> int:
        """Estimate total tokens in batch using message.estimate_tokens()."""
        return sum(msg.estimate_tokens for msg in self.messages)

    @property
    def is_full(self) -> bool:
        logging.error(f"Estimated tokens {self.estimated_tokens}")
        return self.estimated_tokens >= self._batch_token_limit

    @property
    def size(self) -> int:
        return len(self.messages)


class BatchManager:
    """Manages message batching for ChromaDB storage.

    Accumulates messages in batches and provides batch metadata
    for contextual retrieval. Thread-safe implementation.
    
    Batching is based on token count (default 30000 tokens),
    not message count.
    """

    def __init__(self, batch_token_limit: int | None = None) -> None:
        self._batch_token_limit = batch_token_limit or application_settings.batch_token_limit or 30000
        self._lock = threading.Lock()
        self._current_batch: dict[int, Batch] = {}

    @property
    def batch_token_limit(self) -> int:
        return self._batch_token_limit

    def _generate_batch_id(self) -> str:
        """Generate chronologically ordered unique batch ID.

        Uses ULID for:
        - Lexicographically sortable (chronological order)
        - Globally unique
        - No runtime state dependencies

        Returns:
            ULID string.
        """
        return ulid.ulid()

    def add_message(self, user_id: int, message: Message) -> tuple[Batch | None, bool]:
        """Add message to batch accumulator.

        Creates new batch if needed. Returns current batch and
        whether it's full (ready for archiving).

        Args:
            user_id: The user ID.
            message: Message to add.

        Returns:
            Tuple of (current_batch, is_full).
        """
        with self._lock:
            # Create first batch if needed
            if self._current_batch.get(user_id) is None:
                self._current_batch[user_id] = Batch(
                    batch_id=self._generate_batch_id(),
                    prev_batch_id=None,
                    _batch_token_limit=self._batch_token_limit,
                )

            # Add message to current batch
            self._current_batch[user_id].messages.append(message)
            
            was_full = self._current_batch[user_id].is_full
            
            # If batch just became full, recreate the batch
            if was_full:
                current = self._current_batch[user_id]
                self._current_batch[user_id] = Batch(
                    batch_id=self._generate_batch_id(),
                    prev_batch_id=current.batch_id,
                    _batch_token_limit=self._batch_token_limit,
                )
                return current, True

            return self._current_batch[user_id], False

    def flush(self) -> Batch | None:
        """Flush current incomplete batch.

        Called on:
        - Cache cleanup
        - Summarization
        - Shutdown
        - Timeout flush

        Returns:
            Batch to store (may be None).
        """
        with self._lock:
            batch = self._current_batch
            self._current_batch = None
            return batch

    def get_current_batch(self) -> Batch | None:
        """Get current batch without flushing.

        Returns:
            Current batch or None.
        """
        with self._lock:
            return self._current_batch


def create_user_batch_manager(user_id: int, batch_token_limit: int | None = None) -> BatchManager:
    """Create batch manager for user with thread-safe lock.

    Args:
        user_id: The user ID.
        batch_token_limit: Token limit per batch (default 30000).

    Returns:
        Configured BatchManager instance.
    """
    return BatchManager(batch_token_limit=batch_token_limit)