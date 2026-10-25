"""Unit tests for the UsageCacheStore singleton."""

from chibi.schemas.app import UsageSchema
from chibi.services.usage_cache import UsageCacheStore


class TestUsageCacheStore:
    """Tests for the ephemeral (user_id, thread_id) prompt-size cache."""

    def setup_method(self) -> None:
        """Clear the singleton state before each test for isolation."""
        UsageCacheStore()._data.clear()

    def test_store_and_get_round_trip(self) -> None:
        """Writing a value makes it readable for the same key."""
        usage = UsageSchema(prompt_tokens=123, completion_tokens=10, total_tokens=133)
        UsageCacheStore().store(user_id=42, thread_id=7, usage=usage, provider="OpenAI")

        assert UsageCacheStore().get(user_id=42, thread_id=7) == 123

    def test_unknown_key_returns_none(self) -> None:
        """Reading a key that was never written returns None, not 0."""
        assert UsageCacheStore().get(user_id=1, thread_id=0) is None

    def test_overwrite_keeps_last_value(self) -> None:
        """Repeated writes for the same key overwrite with the last value."""
        UsageCacheStore().store(
            user_id=42, thread_id=7, usage=UsageSchema(prompt_tokens=100, total_tokens=100), provider="OpenAI"
        )
        UsageCacheStore().store(
            user_id=42, thread_id=7, usage=UsageSchema(prompt_tokens=200, total_tokens=200), provider="OpenAI"
        )

        assert UsageCacheStore().get(user_id=42, thread_id=7) == 200

    def test_singleton_identity(self) -> None:
        """Two instantiations share the same underlying data."""
        first = UsageCacheStore()
        second = UsageCacheStore()

        assert first is second
        first.store(user_id=1, thread_id=2, usage=UsageSchema(prompt_tokens=999, total_tokens=999), provider="OpenAI")
        assert second.get(user_id=1, thread_id=2) == 999

    def test_anthropic_includes_cache_fields(self) -> None:
        """Anthropic prompt size includes cache creation and cache read tokens."""
        usage = UsageSchema(
            prompt_tokens=100,
            completion_tokens=10,
            cache_creation_input_tokens=25,
            cache_read_input_tokens=15,
            total_tokens=150,
        )
        UsageCacheStore().store(user_id=1, thread_id=0, usage=usage, provider="Anthropic")

        assert UsageCacheStore().get(user_id=1, thread_id=0) == 140

    def test_non_anthropic_uses_prompt_tokens_only(self) -> None:
        """Non-Anthropic providers must not double-count cache fields."""
        usage = UsageSchema(
            prompt_tokens=100,
            completion_tokens=10,
            cache_read_input_tokens=30,
            total_tokens=140,
        )
        UsageCacheStore().store(user_id=1, thread_id=0, usage=usage, provider="OpenAI")

        assert UsageCacheStore().get(user_id=1, thread_id=0) == 100

    def test_provider_name_matching_is_case_insensitive(self) -> None:
        """Provider name comparison is case-insensitive."""
        usage = UsageSchema(
            prompt_tokens=10,
            cache_creation_input_tokens=5,
            cache_read_input_tokens=5,
            total_tokens=30,
        )
        UsageCacheStore().store(user_id=1, thread_id=0, usage=usage, provider="anthropic")

        assert UsageCacheStore().get(user_id=1, thread_id=0) == 20
