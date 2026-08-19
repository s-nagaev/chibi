"""Unit tests for provider utilities."""

import json
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from anthropic.types import Message as AnthropicMessage

from chibi.models import Message
from chibi.schemas.app import UsageSchema
from chibi.services.interface import UserInterface
from chibi.services.providers.utils import get_usage_from_anthropic_response, prepare_system_prompt
from chibi.services.usage_cache import UsageCacheStore


def _make_user() -> Any:
    """Build a minimal user stub for prepare_system_prompt tests."""
    return SimpleNamespace(
        working_dir="/tmp",
        approximate_context_size=lambda thread_id: 999,
        messages=[Message(role="user", content="stale")],
        id=1,
        info="",
        llm_skills={},
    )


def _reset_usage_cache() -> None:
    """Clear the UsageCacheStore singleton between tests."""
    UsageCacheStore()._data.clear()


@pytest.mark.asyncio
async def test_prepare_system_prompt_context_size_na_when_store_empty() -> None:
    """Empty store → approximate_context_size is 'n/a' and no warning is emitted."""
    _reset_usage_cache()
    user = _make_user()
    interface = cast(UserInterface, SimpleNamespace(thread_id=1, uses_uploaded_file_storage=False))
    conversation = [
        Message(role="user", content="Hello there"),
        Message(role="assistant", content="General Kenobi"),
    ]

    with (
        patch("chibi.services.providers.utils.get_chibi_user", new=AsyncMock(return_value=user)),
        patch("chibi.services.providers.utils.get_builtin_skill_names", return_value=[]),
    ):
        prompt_json = await prepare_system_prompt("base", 1, interface, conversation_messages=conversation)

    prompt = json.loads(prompt_json)
    assert prompt["approximate_context_size"] == "n/a"
    assert "context_size_warning" not in prompt


@pytest.mark.asyncio
async def test_prepare_system_prompt_context_size_shows_real_value_and_percentage() -> None:
    """Non-empty store → prompt shows real token count plus percentage of max_history_tokens."""
    _reset_usage_cache()
    user = _make_user()
    interface = cast(UserInterface, SimpleNamespace(thread_id=1, uses_uploaded_file_storage=False))

    store = UsageCacheStore()
    store._data[store._make_key(user_id=1, thread_id=1)] = 58372

    with (
        patch("chibi.services.providers.utils.get_chibi_user", new=AsyncMock(return_value=user)),
        patch("chibi.services.providers.utils.get_builtin_skill_names", return_value=[]),
    ):
        prompt_json = await prepare_system_prompt("base", 1, interface)

    prompt = json.loads(prompt_json)
    from chibi.config import gpt_settings

    max_tokens = gpt_settings.max_history_tokens
    expected_pct = round(58372 / max_tokens * 100)
    assert prompt["approximate_context_size"] == f"58,372 tokens ({expected_pct}% of {max_tokens:,} limit)"
    assert "context_size_warning" not in prompt


@pytest.mark.asyncio
async def test_prepare_system_prompt_warning_fires_above_threshold() -> None:
    """Warning is emitted when the real context exceeds the configured threshold percentage."""
    _reset_usage_cache()
    user = _make_user()
    interface = cast(UserInterface, SimpleNamespace(thread_id=0, uses_uploaded_file_storage=False))

    store = UsageCacheStore()
    store._data[store._make_key(user_id=1, thread_id=0)] = 120000

    with (
        patch("chibi.services.providers.utils.get_chibi_user", new=AsyncMock(return_value=user)),
        patch("chibi.services.providers.utils.get_builtin_skill_names", return_value=[]),
        patch("chibi.config.gpt.gpt_settings.context_size_warning_threshold", 50),
    ):
        prompt_json = await prepare_system_prompt("base", 1, interface)

    prompt = json.loads(prompt_json)
    assert "context_size_warning" in prompt
    assert "50%" in prompt["context_size_warning"]


@pytest.mark.asyncio
async def test_prepare_system_prompt_warning_silent_at_or_below_threshold() -> None:
    """Warning is NOT emitted when the real context is at or below the threshold."""
    _reset_usage_cache()
    user = _make_user()
    interface = cast(UserInterface, SimpleNamespace(thread_id=0, uses_uploaded_file_storage=False))

    store = UsageCacheStore()
    store._data[store._make_key(user_id=1, thread_id=0)] = 100000

    with (
        patch("chibi.services.providers.utils.get_chibi_user", new=AsyncMock(return_value=user)),
        patch("chibi.services.providers.utils.get_builtin_skill_names", return_value=[]),
        patch("chibi.config.gpt.gpt_settings.context_size_warning_threshold", 50),
    ):
        prompt_json = await prepare_system_prompt("base", 1, interface)

    prompt = json.loads(prompt_json)
    assert "context_size_warning" not in prompt


@pytest.mark.asyncio
async def test_prepare_system_prompt_warning_threshold_from_config() -> None:
    """The warning threshold is read from config, not hardcoded."""
    _reset_usage_cache()
    user = _make_user()
    interface = cast(UserInterface, SimpleNamespace(thread_id=0, uses_uploaded_file_storage=False))

    store = UsageCacheStore()
    store._data[store._make_key(user_id=1, thread_id=0)] = 60000

    with (
        patch("chibi.services.providers.utils.get_chibi_user", new=AsyncMock(return_value=user)),
        patch("chibi.services.providers.utils.get_builtin_skill_names", return_value=[]),
        patch("chibi.config.gpt.gpt_settings.context_size_warning_threshold", 25),
    ):
        prompt_json = await prepare_system_prompt("base", 1, interface)

    prompt = json.loads(prompt_json)
    assert "context_size_warning" in prompt
    assert "25%" in prompt["context_size_warning"]


@pytest.mark.asyncio
async def test_prepare_system_prompt_uses_correct_key_matching_write_side() -> None:
    """The read key (user_id, thread_id) matches the write-side convention."""
    _reset_usage_cache()
    user = _make_user()
    interface = cast(UserInterface, SimpleNamespace(thread_id=3, uses_uploaded_file_storage=False))

    store = UsageCacheStore()
    store.store(user_id=1, thread_id=3, usage=UsageSchema(prompt_tokens=12345), provider="openai")

    with (
        patch("chibi.services.providers.utils.get_chibi_user", new=AsyncMock(return_value=user)),
        patch("chibi.services.providers.utils.get_builtin_skill_names", return_value=[]),
    ):
        prompt_json = await prepare_system_prompt("base", 1, interface)

    prompt = json.loads(prompt_json)
    assert "12,345 tokens" in prompt["approximate_context_size"]


def _make_anthropic_response(
    output_tokens: int,
    input_tokens: int,
    cache_creation_input_tokens: Any = None,
    cache_read_input_tokens: Any = None,
) -> AnthropicMessage:
    """Build a minimal Anthropic-style response for usage extraction tests."""
    usage = SimpleNamespace(
        output_tokens=output_tokens,
        input_tokens=input_tokens,
    )
    if cache_creation_input_tokens is not None:
        usage.cache_creation_input_tokens = cache_creation_input_tokens
    if cache_read_input_tokens is not None:
        usage.cache_read_input_tokens = cache_read_input_tokens
    return cast(AnthropicMessage, SimpleNamespace(usage=usage))


def test_get_usage_from_anthropic_response_includes_cache_tokens() -> None:
    """Total tokens is the raw physical sum including both Anthropic cache fields."""
    response = _make_anthropic_response(
        output_tokens=157,
        input_tokens=335,
        cache_creation_input_tokens=14531,
        cache_read_input_tokens=0,
    )

    usage = get_usage_from_anthropic_response(response)

    assert usage == UsageSchema(
        completion_tokens=157,
        prompt_tokens=335,
        cache_creation_input_tokens=14531,
        cache_read_input_tokens=0,
        total_tokens=15023,
    )


def test_get_usage_from_anthropic_response_no_caching() -> None:
    """When cache fields are zero, total tokens reduces to the old output+input sum."""
    response = _make_anthropic_response(
        output_tokens=66,
        input_tokens=784,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )

    usage = get_usage_from_anthropic_response(response)

    assert usage == UsageSchema(
        completion_tokens=66,
        prompt_tokens=784,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        total_tokens=850,
    )


def test_get_usage_from_anthropic_response_missing_cache_fields() -> None:
    """Missing or None cache attributes are treated as zero without raising."""
    response = _make_anthropic_response(
        output_tokens=10,
        input_tokens=20,
        cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
    )

    usage = get_usage_from_anthropic_response(response)

    assert usage == UsageSchema(
        completion_tokens=10,
        prompt_tokens=20,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        total_tokens=30,
    )
