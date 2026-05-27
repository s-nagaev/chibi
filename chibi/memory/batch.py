"""Batch manager for accumulating messages before ChromaDB storage."""
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TypedDict

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
    _batch_size: int = 10

    @property
    def is_full(self) -> bool:
        return len(self.messages) >= self._batch_size

    @property
    def size(self) -> int:
        return len(self.messages)


class BatchManager:
    """Manages message batching for ChromaDB storage.

    Accumulates messages in batches and provides batch metadata
    for contextual retrieval. Thread-safe implementation.
    """

    def __init__(self, batch_size: int | None = None) -> None:
        self._batch_size = batch_size or application_settings.batch_size or 10
        self._lock = threading.Lock()
        self._current_batch: Batch | None = None

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def _generate_batch_id(self) -> str:
        """Generate globally unique batch ID.
        
        Uses UUID for:
        - Globally unique
        - No runtime state dependencies
        
        Returns:
            UUID string.
        """
        return str(uuid.uuid4())

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
            if self._current_batch is None:
                self._current_batch = Batch(
                    batch_id=self._generate_batch_id(),
                    prev_batch_id=None,
                    _batch_size=self._batch_size,
                )

            # Add message to current batch
            self._current_batch.messages.append(message)
            
            was_full = self._current_batch.is_full
            
            # If batch just became full, create next batch
            if was_full:
                current = self._current_batch
                self._current_batch = Batch(
                    batch_id=self._generate_batch_id(),
                    prev_batch_id=current.batch_id,
                    _batch_size=self._batch_size,
                )
                return current, True

            return self._current_batch, False

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


def create_user_batch_manager(user_id: int, batch_size: int | None = None) -> BatchManager:
    """Create batch manager for user with thread-safe lock.

    Args:
        user_id: The user ID.
        batch_size: Size of batch.

    Returns:
        Configured BatchManager instance.
    """
    return BatchManager(batch_size=batch_size)