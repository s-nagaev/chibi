"""In-memory store for the last provider-reported real prompt token size."""

from chibi.schemas.app import UsageSchema
from chibi.utils.app import SingletonMeta


class UsageCacheStore(metaclass=SingletonMeta):
    """Process-lifetime cache of the real prompt size per (user_id, thread_id).

    The store is intentionally ephemeral: a process restart yields an empty cache.
    """

    def __init__(self) -> None:
        """Initialize the usage cache store."""
        self._data: dict[str, int] = {}

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

        Repeated writes for the same key overwrite the previous value (last-write-wins).

        Args:
            user_id: The unique user identifier.
            thread_id: The message thread identifier (0 for the main thread).
            usage: Provider usage schema.
            provider: Provider name.
        """
        self._data[self._make_key(user_id=user_id, thread_id=thread_id)] = self._raw_prompt_tokens(
            usage=usage, provider=provider
        )

    def get(self, user_id: int, thread_id: int) -> int | None:
        """Fetch the last real prompt size for the given conversation key.

        Args:
            user_id: The unique user identifier.
            thread_id: The message thread identifier (0 for the main thread).

        Returns:
            The stored token count, or None if the key has never been written.
        """
        return self._data.get(self._make_key(user_id=user_id, thread_id=thread_id))
