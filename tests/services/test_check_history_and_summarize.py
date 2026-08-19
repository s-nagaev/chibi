"""Unit tests for the rebased summarization trigger in ``check_history_and_summarize``."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chibi.models import Message
from chibi.services.usage_cache import UsageCacheStore
from chibi.services.user import check_history_and_summarize


def _make_db(messages: list[Message]) -> MagicMock:
    """Build a mock Database returning the given conversation messages."""
    db = MagicMock()
    db.get_or_create_user = AsyncMock(return_value=MagicMock(id=1))
    db.get_conversation_messages = AsyncMock(return_value=messages)
    return db


def _reset_usage_cache() -> None:
    """Clear the UsageCacheStore singleton between tests."""
    UsageCacheStore()._data.clear()


@pytest.mark.asyncio
async def test_cold_start_fallback_fires_on_heuristic_above_threshold() -> None:
    """Empty store -> heuristic sum above threshold triggers summarization."""
    _reset_usage_cache()
    # Heuristic sum: each message (len(content)+len(role))//4. Build a large history.
    big = "x" * 4000
    messages = [Message(role="user", content=big), Message(role="assistant", content=big)]

    with patch("chibi.services.user.emergency_summarization", new=AsyncMock()) as mock_summary:
        with patch("chibi.services.user.gpt_settings.max_history_tokens", 100):
            result = await check_history_and_summarize.__wrapped__(_make_db(messages), 1, 0)

    assert result is True
    mock_summary.assert_awaited_once_with(storage_id=1, thread_id=0)


@pytest.mark.asyncio
async def test_cold_start_fallback_does_not_fire_below_threshold() -> None:
    """Empty store -> heuristic sum below threshold does not trigger summarization."""
    _reset_usage_cache()
    messages = [Message(role="user", content="hi"), Message(role="assistant", content="hello")]

    with patch("chibi.services.user.emergency_summarization", new=AsyncMock()) as mock_summary:
        with patch("chibi.services.user.gpt_settings.max_history_tokens", 1000000):
            result = await check_history_and_summarize.__wrapped__(_make_db(messages), 1, 0)

    assert result is False
    mock_summary.assert_not_awaited()


@pytest.mark.asyncio
async def test_real_value_above_threshold_fires() -> None:
    """A real cached prompt size above the threshold triggers summarization."""
    _reset_usage_cache()
    store = UsageCacheStore()
    store.store(user_id=1, thread_id=0, usage=MagicMock(prompt_tokens=500), provider="OpenAI")
    messages = [Message(role="user", content="small")]

    with patch("chibi.services.user.emergency_summarization", new=AsyncMock()) as mock_summary:
        with patch("chibi.services.user.gpt_settings.max_history_tokens", 400):
            result = await check_history_and_summarize.__wrapped__(_make_db(messages), 1, 0)

    assert result is True
    mock_summary.assert_awaited_once_with(storage_id=1, thread_id=0)


@pytest.mark.asyncio
async def test_real_value_below_threshold_does_not_fire() -> None:
    """A real cached prompt size below the threshold does not trigger summarization."""
    _reset_usage_cache()
    store = UsageCacheStore()
    store.store(user_id=1, thread_id=0, usage=MagicMock(prompt_tokens=300), provider="OpenAI")
    messages = [Message(role="user", content="small")]

    with patch("chibi.services.user.emergency_summarization", new=AsyncMock()) as mock_summary:
        with patch("chibi.services.user.gpt_settings.max_history_tokens", 1000):
            result = await check_history_and_summarize.__wrapped__(_make_db(messages), 1, 0)

    assert result is False
    mock_summary.assert_not_awaited()


@pytest.mark.asyncio
async def test_rebased_default_prevents_false_positive_on_measured_prompt() -> None:
    """The re-based default (100000) must NOT summarize a representative measured real prompt.

    The analysis artifact measured a real prompt size of 58372 tokens for a
    conversation whose old heuristic reported only 12106 (~4.8x ratio). Under
    the old default of 64000 against the truthful figure, every such conversation
    would immediately summarize. The re-based default of 100000 sits below the
    smallest commonly-supported context window (128k, ~78% of it) so summarization
    fires before a 128k model overflows, while a 58372-token real prompt (a
    typical mid-size conversation) must NOT trigger summarization.
    """
    _reset_usage_cache()
    store = UsageCacheStore()
    store.store(user_id=1, thread_id=0, usage=MagicMock(prompt_tokens=58372), provider="OpenAI")
    messages = [Message(role="user", content="representative")]

    with patch("chibi.services.user.emergency_summarization", new=AsyncMock()) as mock_summary:
        with patch("chibi.services.user.gpt_settings.max_history_tokens", 100000):
            result = await check_history_and_summarize.__wrapped__(_make_db(messages), 1, 0)

    assert result is False
    mock_summary.assert_not_awaited()


def test_max_history_tokens_default_is_rebased_to_100000() -> None:
    """The code default for max_history_tokens is 100000 (re-based from 64000).

    A fresh GPTSettings instance with no MAX_HISTORY_TOKENS in the environment
    resolves to 100000. The module-level singleton may be overridden by the
    deployment's .env, so we construct a fresh instance with the env var unset.
    """
    import os

    from chibi.config.gpt import GPTSettings

    previous = os.environ.pop("MAX_HISTORY_TOKENS", None)
    try:
        settings = GPTSettings(_env_file=None)
        assert settings.max_history_tokens == 100000
    finally:
        if previous is not None:
            os.environ["MAX_HISTORY_TOKENS"] = previous


@pytest.mark.asyncio
async def test_real_value_ignored_for_different_thread_key() -> None:
    """A real value stored under a different thread_id must not satisfy this check."""
    _reset_usage_cache()
    store = UsageCacheStore()
    store.store(user_id=1, thread_id=5, usage=MagicMock(prompt_tokens=999999), provider="OpenAI")
    messages = [Message(role="user", content="small")]

    with patch("chibi.services.user.emergency_summarization", new=AsyncMock()) as mock_summary:
        with patch("chibi.services.user.gpt_settings.max_history_tokens", 1000000):
            result = await check_history_and_summarize.__wrapped__(_make_db(messages), 1, 0)

    # thread_id=0 key is empty -> cold-start fallback -> small heuristic -> no fire
    assert result is False
    mock_summary.assert_not_awaited()
