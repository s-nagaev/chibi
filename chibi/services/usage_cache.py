"""In-memory store for the last provider-reported real prompt token size."""

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
