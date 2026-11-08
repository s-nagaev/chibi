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

    def test_unreported_usage_does_not_overwrite_known_good_value(self) -> None:
        """A zero-computed usage (provider reported nothing) keeps the previous value."""
        UsageCacheStore().store(
            user_id=42, thread_id=7, usage=UsageSchema(prompt_tokens=150, total_tokens=160), provider="OpenAI"
        )
        UsageCacheStore().store(user_id=42, thread_id=7, usage=UsageSchema(), provider="OpenAI")

        assert UsageCacheStore().get(user_id=42, thread_id=7) == 150
        assert UsageCacheStore().is_unreported(user_id=42, thread_id=7) is False

    def test_unreported_usage_is_stored_explicitly_when_no_prior_value(self) -> None:
        """With no prior entry, an unreported usage is stored as the explicit marker."""
        UsageCacheStore().store(user_id=42, thread_id=7, usage=UsageSchema(), provider="OpenAI")

        assert UsageCacheStore().get(user_id=42, thread_id=7) is None
        assert UsageCacheStore().is_unreported(user_id=42, thread_id=7) is True

    def test_absent_key_is_not_unreported(self) -> None:
        """A key that was never written is neither measured nor marked unreported."""
        assert UsageCacheStore().is_unreported(user_id=42, thread_id=7) is False

    def test_real_value_after_unreported_marker_overwrites(self) -> None:
        """A later real measurement replaces the explicit unreported marker."""
        UsageCacheStore().store(user_id=42, thread_id=7, usage=UsageSchema(), provider="OpenAI")
        assert UsageCacheStore().is_unreported(user_id=42, thread_id=7) is True

        UsageCacheStore().store(
            user_id=42, thread_id=7, usage=UsageSchema(prompt_tokens=90, total_tokens=100), provider="OpenAI"
        )

        assert UsageCacheStore().get(user_id=42, thread_id=7) == 90
        assert UsageCacheStore().is_unreported(user_id=42, thread_id=7) is False

    def test_anthropic_all_zero_cache_fields_still_unreported(self) -> None:
        """An Anthropic usage summing to zero (no cache parts either) is unreported."""
        UsageCacheStore().store(
            user_id=42,
            thread_id=7,
            usage=UsageSchema(cache_creation_input_tokens=0, cache_read_input_tokens=0),
            provider="Anthropic",
        )

        assert UsageCacheStore().is_unreported(user_id=42, thread_id=7) is True

    def test_invalidate_drops_stored_value(self) -> None:
        """Invalidation removes the cached value for the key."""
        UsageCacheStore().store(
            user_id=42, thread_id=7, usage=UsageSchema(prompt_tokens=150, total_tokens=160), provider="OpenAI"
        )
        UsageCacheStore().invalidate(user_id=42, thread_id=7)

        assert UsageCacheStore().get(user_id=42, thread_id=7) is None

    def test_invalidate_drops_unreported_marker(self) -> None:
        """Invalidation also clears the explicit unreported marker."""
        UsageCacheStore().store(user_id=42, thread_id=7, usage=UsageSchema(), provider="OpenAI")
        UsageCacheStore().invalidate(user_id=42, thread_id=7)

        assert UsageCacheStore().get(user_id=42, thread_id=7) is None
        assert UsageCacheStore().is_unreported(user_id=42, thread_id=7) is False

    def test_invalidate_unknown_key_does_not_raise(self) -> None:
        """Invalidating a key that has no entry yet (or ever) is a tolerated no-op."""
        UsageCacheStore().invalidate(user_id=1, thread_id=0)

        assert UsageCacheStore().get(user_id=1, thread_id=0) is None

    def test_raw_prompt_tokens_is_the_shared_unreported_signal(self) -> None:
        """The raw-size computation is the single "usage reported" signal.

        The durable per-thread measurement writer
        (``chibi.services.context_measurement.persist_measured_usage``) reuses
        this exact computation: a raw size of 0 means "provider reported no
        usage" for both the in-memory store and the durable record, so the two
        can never disagree about which responses carry a measurement.
        """
        assert UsageCacheStore._raw_prompt_tokens(usage=UsageSchema(), provider="OpenAI") == 0
        anthropic_usage = UsageSchema(
            prompt_tokens=100,
            cache_creation_input_tokens=25,
            cache_read_input_tokens=15,
        )
        assert UsageCacheStore._raw_prompt_tokens(usage=anthropic_usage, provider="Anthropic") == 140
        assert UsageCacheStore._raw_prompt_tokens(usage=anthropic_usage, provider="OpenAI") == 100
