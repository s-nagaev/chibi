"""In-memory side-channel recording the context guard's trim events per thread.

The pre-call guard (:func:`chibi.services.user._enforce_context_budget`) and
the in-loop guard (:func:`chibi.services.user.enforce_in_loop_budget`) shrink
an oversized request transiently: storage is never touched and the shrink is
silent, so the model receives a request whose oldest parts are gone without a
word. This store is the write side of making that loss visible: a guard that
actually shrinks a request records how many units it dropped and how many
messages it truncated, keyed by the same ``(storage_id, thread_id)`` pair the
rest of the context machinery uses. The read side is
:func:`chibi.services.providers.utils.collect_volatile_context_note`, which
consumes the record for the conversation and turns it into one
``guard_trim_warning`` line of the volatile ``<context_note>`` block.

The store is intentionally ephemeral (UsageCacheStore-style, same process):
a restart yields an empty store, storage is never touched, and a consumed
record is gone, so the warning line appears only when a trim actually happened
since the previous note, never on every request.
"""

from dataclasses import dataclass

from chibi.utils.app import SingletonMeta


@dataclass
class TrimStats:
    """The trim events accumulated for one conversation since the last read.

    Attributes:
        dropped_units: Complete units (a message, or an assistant tool_calls
            message with its tool results) the guard removed from the request.
        dropped_messages: Individual messages removed; a dropped unit can
            carry more than one message, so this count is greater than or
            equal to ``dropped_units``.
        truncated_messages: Messages whose ``content`` was shortened in place
            (pre-call guard only; the in-loop guard is drop-only).
    """

    dropped_units: int = 0
    dropped_messages: int = 0
    truncated_messages: int = 0

    @property
    def is_empty(self) -> bool:
        """Tell a no-op record apart from a real trim.

        Returns:
            True when no drop and no truncation was recorded.
        """
        return not (self.dropped_units or self.dropped_messages or self.truncated_messages)

    def add(self, other: "TrimStats") -> None:
        """Accumulate another stats bundle into this one.

        Args:
            other: The bundle to merge in; it is left unchanged.
        """
        self.dropped_units += other.dropped_units
        self.dropped_messages += other.dropped_messages
        self.truncated_messages += other.truncated_messages


class ContextTrimStore(metaclass=SingletonMeta):
    """Process-lifetime record of context-guard trim events per conversation.

    The store is intentionally ephemeral: a process restart yields an empty
    store. Records accumulate until consumed, so several trims of one turn
    (an in-loop trim after a pre-call trim) are reported together, and a
    consumed record is gone, which keeps the warning off requests that were
    not trimmed.

    Safe under the process's single asyncio event loop: every call site is
    synchronous, await-free code on the loop (the read-modify-write in
    ``record`` and the ``dict.pop`` in ``consume`` need no lock under this
    assumption); not safe if ever called from executor threads.
    """

    def __init__(self) -> None:
        """Initialize the trim store."""
        self._data: dict[str, TrimStats] = {}

    @staticmethod
    def _make_key(storage_id: int, thread_id: int) -> str:
        """Build the internal storage key for a (storage_id, thread_id) pair.

        The key matches the one the guards use for measurement
        (``user_id for private chats, chat_id for groups``), so the reader
        resolves the exact conversation whose request was trimmed.

        Args:
            storage_id: The storage key of the conversation (user_id for
                private chats, chat_id for groups).
            thread_id: The message thread identifier (0 for the main thread).

        Returns:
            Internal dictionary key.
        """
        return f"{storage_id}_{thread_id}"

    def record(
        self,
        storage_id: int,
        thread_id: int,
        dropped_units: int = 0,
        dropped_messages: int = 0,
        truncated_messages: int = 0,
    ) -> None:
        """Accumulate one guard trim event for the conversation.

        Guards call this only when their shrink actually removed or shortened
        something; a zero-valued event (the shrink ran but changed nothing,
        which happens on its internal error paths) is ignored, so a record
        always implies a real trim.

        Args:
            storage_id: The storage key of the conversation (user_id for
                private chats, chat_id for groups).
            thread_id: The message thread identifier (0 for the main thread).
            dropped_units: Complete units removed from the request.
            dropped_messages: Individual messages removed.
            truncated_messages: Messages whose content was shortened in place.
        """
        if not (dropped_units or dropped_messages or truncated_messages):
            return
        stats = self._data.setdefault(self._make_key(storage_id=storage_id, thread_id=thread_id), TrimStats())
        stats.dropped_units += dropped_units
        stats.dropped_messages += dropped_messages
        stats.truncated_messages += truncated_messages

    def consume(self, storage_id: int, thread_id: int) -> TrimStats | None:
        """Return the accumulated trim stats for the conversation and clear them.

        The read-once semantics belong to the note contract: the warning line
        must describe the trims that happened since the last disclosure, and
        must not repeat on every subsequent request.

        Args:
            storage_id: The storage key of the conversation (user_id for
                private chats, chat_id for groups).
            thread_id: The message thread identifier (0 for the main thread).

        Returns:
            The accumulated stats, or None when nothing was recorded since
            the last consume.
        """
        return self._data.pop(self._make_key(storage_id=storage_id, thread_id=thread_id), None)

    def clear(self, storage_id: int, thread_id: int) -> None:
        """Drop any accumulated stats for the conversation without reading them.

        Tolerant of a key that has no entry; never raises on absence.

        Args:
            storage_id: The storage key of the conversation (user_id for
                private chats, chat_id for groups).
            thread_id: The message thread identifier (0 for the main thread).
        """
        self._data.pop(self._make_key(storage_id=storage_id, thread_id=thread_id), None)
