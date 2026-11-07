"""In-memory stores for per-thread ephemeral state.

:class:`UsageCacheStore` holds the last provider-reported real prompt token
size; :class:`CompactionStateStore` holds the throttle state of the forced
post-delivery compaction trigger (the in-flight flag and the back-off
bookkeeping). Both are intentionally ephemeral: a process restart yields an
empty store.
"""

import time

from chibi.schemas.app import UsageSchema
from chibi.utils.app import SingletonMeta


class UsageCacheStore(metaclass=SingletonMeta):
    """Process-lifetime cache of the real prompt size per (user_id, thread_id).

    The store is intentionally ephemeral: a process restart yields an empty cache.
    """

    def __init__(self) -> None:
        """Initialize the usage cache store."""
        self._data: dict[str, int | None] = {}

    @staticmethod
    def _make_key(user_id: int, thread_id: int) -> str:
        """Build the internal storage key for a (user_id, thread_id) pair.

        Args:
            user_id: The unique user identifier.
            thread_id: The message thread identifier (0 for the main thread).

        Returns:
            Internal dictionary key.
        """
        return f"{user_id}_{thread_id}"

    @staticmethod
    def _raw_prompt_tokens(usage: UsageSchema, provider: str) -> int:
        """Compute the true raw input token count from provider usage data.

        Anthropic reports cached input separately from ``input_tokens``, so the
        cached parts must be added to obtain the physical prompt size. All other
        providers include cached tokens inside their prompt/input token count.

        Args:
            usage: Provider usage schema.
            provider: Provider name.

        Returns:
            The raw number of input tokens processed by the provider.
        """
        if provider.lower() == "anthropic":
            return usage.prompt_tokens + usage.cache_creation_input_tokens + usage.cache_read_input_tokens
        return usage.prompt_tokens

    def store(self, user_id: int, thread_id: int, usage: UsageSchema, provider: str) -> None:
        """Remember the last real prompt size for the given conversation key.

        Repeated writes for the same key overwrite the previous value
        (last-write-wins), except when the provider reports no usable usage:
        a computed raw size of zero cannot be a measurement of a real
        non-empty request, so it is recorded as the explicit "unreported"
        marker (``None``) instead of a zero, and an existing known-good value
        is never replaced by it. Readers can therefore distinguish a key that
        was never measured (absent), a turn whose provider reported nothing
        (the ``None`` marker, see :meth:`is_unreported`), and a real
        measurement (a positive integer).

        Args:
            user_id: The unique user identifier.
            thread_id: The message thread identifier (0 for the main thread).
            usage: Provider usage schema.
            provider: Provider name.
        """
        raw = self._raw_prompt_tokens(usage=usage, provider=provider)
        key = self._make_key(user_id=user_id, thread_id=thread_id)
        if raw == 0:
            if key not in self._data:
                self._data[key] = None
            return
        self._data[key] = raw

    def get(self, user_id: int, thread_id: int) -> int | None:
        """Fetch the last real prompt size for the given conversation key.

        Args:
            user_id: The unique user identifier.
            thread_id: The message thread identifier (0 for the main thread).

        Returns:
            The stored token count, or None when the key has never carried a
            real measurement (never written, or the provider last reported
            no usage, check :meth:`is_unreported` to tell the two apart).
        """
        return self._data.get(self._make_key(user_id=user_id, thread_id=thread_id))

    def is_unreported(self, user_id: int, thread_id: int) -> bool:
        """Tell a turn with no provider-reported usage apart from an absent entry.

        Args:
            user_id: The unique user identifier.
            thread_id: The message thread identifier (0 for the main thread).

        Returns:
            True when the last stored fact for the key is the explicit
            "provider reported no usage" marker; False when the key holds a
            real measurement or has never been written.
        """
        return self._data.get(self._make_key(user_id=user_id, thread_id=thread_id), 0) is None

    def invalidate(self, user_id: int, thread_id: int) -> None:
        """Drop the cached prompt size for the given conversation key.

        This is the single invalidation entry point for every path that
        resets, replaces or forks a thread's history (``/reset``, thread
        cloning, ``summarize_history``, ``clear_tool_call_history`` and the
        emergency summarization): the stored value reflects the pre-reset
        prompt and would immediately re-trigger summarization or misreport
        the context size on the next turn. The call is tolerant of a key
        that has no entry yet (including a per-thread record that a separate
        work item may add later): it never raises on absence.

        Args:
            user_id: The unique user identifier.
            thread_id: The message thread identifier (0 for the main thread).
        """
        self._data.pop(self._make_key(user_id=user_id, thread_id=thread_id), None)


class CompactionStateStore(metaclass=SingletonMeta):
    """Process-lifetime throttle state of the forced compaction per (user_id, thread_id).

    The forced post-delivery compaction trigger uses this store for three
    facts about a conversation key:

    - the per-thread in-flight flag, set before the summarizer call and
      cleared on completion or failure, so a compaction can never overlap
      itself;
    - a per-thread turn counter, advanced once per post-delivery check, so a
      back-off can be expressed in turns;
    - the turn number and the monotonic timestamp of the last FAILED
      compaction attempt, so the trigger stays quiet until three turns have
      passed or thirty minutes have elapsed, whichever first.

    Like :class:`UsageCacheStore`, the store is intentionally ephemeral: a
    process restart clears every flag and forgets every back-off, which is
    acceptable for a throttle (worst case, one extra compaction attempt).
    """

    def __init__(self) -> None:
        """Initialize the empty compaction state store."""
        self._in_flight: set[str] = set()
        self._turn_counters: dict[str, int] = {}
        self._failure_turns: dict[str, int] = {}
        self._failure_monotonic: dict[str, float] = {}

    @staticmethod
    def _make_key(user_id: int, thread_id: int) -> str:
        """Build the internal storage key for a (user_id, thread_id) pair.

        Args:
            user_id: The unique user identifier.
            thread_id: The message thread identifier (0 for the main thread).

        Returns:
            Internal dictionary key.
        """
        return f"{user_id}_{thread_id}"

    def is_in_flight(self, user_id: int, thread_id: int) -> bool:
        """Tell whether a forced compaction for the key is currently running.

        Args:
            user_id: The unique user identifier.
            thread_id: The message thread identifier (0 for the main thread).

        Returns:
            True while the flag set before the summarizer call has not been
            cleared yet.
        """
        return self._make_key(user_id=user_id, thread_id=thread_id) in self._in_flight

    def mark_in_flight(self, user_id: int, thread_id: int) -> None:
        """Set the per-thread in-flight flag before the summarizer call runs.

        Args:
            user_id: The unique user identifier.
            thread_id: The message thread identifier (0 for the main thread).
        """
        self._in_flight.add(self._make_key(user_id=user_id, thread_id=thread_id))

    def clear_in_flight(self, user_id: int, thread_id: int) -> None:
        """Clear the per-thread in-flight flag on completion or failure.

        The call is tolerant of a key that was never marked.

        Args:
            user_id: The unique user identifier.
            thread_id: The message thread identifier (0 for the main thread).
        """
        self._in_flight.discard(self._make_key(user_id=user_id, thread_id=thread_id))

    def record_turn(self, user_id: int, thread_id: int) -> int:
        """Advance the per-thread turn counter by one post-delivery check.

        Called once per delivered answer, before the trigger conditions are
        evaluated, so a back-off recorded through :meth:`record_failure` can
        be measured in turns.

        Args:
            user_id: The unique user identifier.
            thread_id: The message thread identifier (0 for the main thread).

        Returns:
            The turn number of the current check (1 for the first check of
            the process lifetime).
        """
        key = self._make_key(user_id=user_id, thread_id=thread_id)
        self._turn_counters[key] = self._turn_counters.get(key, 0) + 1
        return self._turn_counters[key]

    def record_failure(self, user_id: int, thread_id: int) -> None:
        """Remember the current turn and time as a failed compaction attempt.

        Args:
            user_id: The unique user identifier.
            thread_id: The message thread identifier (0 for the main thread).
        """
        key = self._make_key(user_id=user_id, thread_id=thread_id)
        self._failure_turns[key] = self._turn_counters.get(key, 0)
        self._failure_monotonic[key] = time.monotonic()

    def is_backed_off(
        self,
        user_id: int,
        thread_id: int,
        turns: int = 3,
        minutes: float = 30,
    ) -> bool:
        """Tell whether the key is still inside the back-off after a failure.

        The back-off expires as soon as EITHER the required number of turns
        have passed since the failed attempt OR the required wall-clock time
        has elapsed (whichever first). A key without a recorded failure is
        never backed off.

        Args:
            user_id: The unique user identifier.
            thread_id: The message thread identifier (0 for the main thread).
            turns: Number of turns that must pass before a retry.
            minutes: Wall-clock minutes that must pass before a retry.

        Returns:
            True while both the turn condition and the time condition still
            hold; False once either expires or no failure was recorded.
        """
        key = self._make_key(user_id=user_id, thread_id=thread_id)
        if key not in self._failure_turns:
            return False
        turns_since_failure = self._turn_counters.get(key, 0) - self._failure_turns[key]
        elapsed_seconds = time.monotonic() - self._failure_monotonic[key]
        return turns_since_failure < turns and elapsed_seconds < minutes * 60
