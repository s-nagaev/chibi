"""Unit tests for provider utilities."""

import json
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from anthropic.types import Message as AnthropicMessage

from chibi.config import application_settings
from chibi.config.app import ClientType
from chibi.constants import SCHEDULER_HINT_PROMPT
from chibi.models import Message, User
from chibi.schemas.app import UsageSchema
from chibi.services.interface import UserInterface
from chibi.services.providers.utils import (
    _is_tool_response_message,
    get_usage_from_anthropic_response,
    prepare_system_prompt,
)
from chibi.services.usage_cache import UsageCacheStore


def _make_user() -> Any:
    """Build a minimal user for prepare_system_prompt tests (real resolution chain)."""
    return User(id=1, working_dir="/tmp", info="", llm_skills={})


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
        patch("chibi.services.providers.utils.get_available_skills", return_value=[]),
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
        patch("chibi.services.providers.utils.get_available_skills", return_value=[]),
        patch("chibi.config.gpt.gpt_settings.max_history_tokens", 200000),
        patch("chibi.config.gpt.gpt_settings.context_size_warning_threshold", 50),
    ):
        prompt_json = await prepare_system_prompt("base", 1, interface)

    prompt = json.loads(prompt_json)
    expected_pct = round(58372 / 200000 * 100)
    assert prompt["approximate_context_size"] == f"58,372 tokens ({expected_pct}% of {200000:,} limit)"
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
        patch("chibi.services.providers.utils.get_available_skills", return_value=[]),
        patch("chibi.config.gpt.gpt_settings.context_size_warning_threshold", 50),
    ):
        prompt_json = await prepare_system_prompt("base", 1, interface)

    prompt = json.loads(prompt_json)
    assert "context_size_warning" in prompt
    assert "50%" in prompt["context_size_warning"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stored_tokens", [50000, 40000])
async def test_prepare_system_prompt_warning_silent_at_or_below_threshold(stored_tokens: int) -> None:
    """Warning is NOT emitted when the real context is at or below the threshold.

    50000 tokens equals exactly 50% of the pinned limit (boundary of the strict ``>`` check),
    40000 tokens is clearly below it.
    """
    _reset_usage_cache()
    user = _make_user()
    interface = cast(UserInterface, SimpleNamespace(thread_id=0, uses_uploaded_file_storage=False))

    store = UsageCacheStore()
    store._data[store._make_key(user_id=1, thread_id=0)] = stored_tokens

    with (
        patch("chibi.services.providers.utils.get_chibi_user", new=AsyncMock(return_value=user)),
        patch("chibi.services.providers.utils.get_available_skills", return_value=[]),
        patch("chibi.config.gpt.gpt_settings.max_history_tokens", 100000),
        patch("chibi.config.gpt.gpt_settings.context_size_warning_threshold", 50),
    ):
        prompt_json = await prepare_system_prompt("base", 1, interface)

    prompt = json.loads(prompt_json)
    expected_pct = round(stored_tokens / 100000 * 100)
    assert expected_pct <= 50
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
        patch("chibi.services.providers.utils.get_available_skills", return_value=[]),
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
        patch("chibi.services.providers.utils.get_available_skills", return_value=[]),
    ):
        prompt_json = await prepare_system_prompt("base", 1, interface)

    prompt = json.loads(prompt_json)
    assert "12,345 tokens" in prompt["approximate_context_size"]


@pytest.fixture
def restore_client_setting() -> Iterator[None]:
    """Restore the original client value after the test."""
    original = application_settings.client
    yield
    application_settings.client = original


@pytest.mark.asyncio
@pytest.mark.usefixtures("restore_client_setting")
@pytest.mark.parametrize("client", ["telegram", "tui", "vscode", "pycharm", "neovim"])
async def test_prepare_system_prompt_includes_active_client(client: ClientType) -> None:
    """The assembled prompt payload carries the active client on every launch path."""
    _reset_usage_cache()
    user = _make_user()
    interface = cast(UserInterface, SimpleNamespace(thread_id=1, uses_uploaded_file_storage=False))
    application_settings.client = client

    with (
        patch("chibi.services.providers.utils.get_chibi_user", new=AsyncMock(return_value=user)),
        patch("chibi.services.providers.utils.get_available_skills", return_value=[]),
    ):
        prompt_json = await prepare_system_prompt("base", 1, interface)

    prompt = json.loads(prompt_json)
    assert prompt["client"] == client


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


def _scheduler_tool_settings(scheduler_tool_enabled: bool, client: str) -> SimpleNamespace:
    """Build a settings namespace for the scheduler tools registration gate."""
    return SimpleNamespace(scheduler_tool_enabled=scheduler_tool_enabled, client=client)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scheduler_tool_enabled", "client", "expected"),
    [
        (True, "telegram", True),
        (True, "vscode", True),
        (True, "terminal", False),
        (False, "telegram", False),
    ],
)
async def test_prepare_system_prompt_scheduler_hint_follows_tool_gate(
    scheduler_tool_enabled: bool, client: str, expected: bool
) -> None:
    """The hint is injected only when the scheduler tool gate + client runner allow the tools."""
    _reset_usage_cache()
    user = _make_user()
    interface = cast(UserInterface, SimpleNamespace(thread_id=1, uses_uploaded_file_storage=False))

    with (
        patch("chibi.services.providers.utils.get_chibi_user", new=AsyncMock(return_value=user)),
        patch("chibi.services.providers.utils.get_available_skills", return_value=[]),
        patch(
            "chibi.services.providers.tools.scheduler.application_settings",
            _scheduler_tool_settings(scheduler_tool_enabled, client),
        ),
    ):
        prompt_json = await prepare_system_prompt("base", 1, interface)

    prompt = json.loads(prompt_json)
    hint_in_prompt = SCHEDULER_HINT_PROMPT in prompt["system_prompt"]
    assert hint_in_prompt is expected


def test_scheduler_hint_prompt_is_compact() -> None:
    """The hint must stay a compact block (at most 4 non-empty lines)."""
    content_lines = [line for line in SCHEDULER_HINT_PROMPT.strip().splitlines() if line.strip()]
    assert 1 <= len(content_lines) <= 4
    assert "schedule_task" in SCHEDULER_HINT_PROMPT
    assert "max_fires" in SCHEDULER_HINT_PROMPT


def _make_tool_response_blob(tool_name: str, filler: str) -> Message:
    """Build a shape-(ii) background tool-response turn (role="user" JSON blob).

    Args:
        tool_name: Tool name stored in the blob payload.
        filler: Payload content controlling the message size.

    Returns:
        The constructed message.
    """
    payload = {"type": "tool response", "tool_name": tool_name, "result": filler}
    return Message(role="user", content=json.dumps(payload))


@pytest.mark.asyncio
async def test_prepare_system_prompt_mandatory_warning_is_imperative() -> None:
    """The context warning text must be imperative and keep the '{threshold}%' substring."""
    _reset_usage_cache()
    user = _make_user()
    interface = cast(UserInterface, SimpleNamespace(thread_id=0, uses_uploaded_file_storage=False))

    store = UsageCacheStore()
    store._data[store._make_key(user_id=1, thread_id=0)] = 120000

    with (
        patch("chibi.services.providers.utils.get_chibi_user", new=AsyncMock(return_value=user)),
        patch("chibi.services.providers.utils.get_available_skills", return_value=[]),
        patch("chibi.config.gpt.gpt_settings.context_size_warning_threshold", 50),
    ):
        prompt_json = await prepare_system_prompt("base", 1, interface)

    prompt = json.loads(prompt_json)
    warning = prompt["context_size_warning"]
    assert "MANDATORY" in warning
    assert "MUST" in warning
    assert "summarize_history" in warning
    assert "clear_tool_call_history" in warning
    assert "50%" in warning


@pytest.mark.asyncio
async def test_prepare_system_prompt_tool_results_warning_fires_above_share_threshold() -> None:
    """The volume-based warning fires when shape-(ii) blobs exceed 25% of the real context size."""
    _reset_usage_cache()
    user = _make_user()
    interface = cast(UserInterface, SimpleNamespace(thread_id=0, uses_uploaded_file_storage=False))

    blob = _make_tool_response_blob("web_search", "SECRET_BLOB_CONTENT" + "x" * 1200)
    conversation = [blob, Message(role="assistant", content="ok")]
    tool_results_tokens = sum(msg.estimate_tokens for msg in conversation if _is_tool_response_message(msg))

    store = UsageCacheStore()
    store._data[store._make_key(user_id=1, thread_id=0)] = tool_results_tokens * 3

    with (
        patch("chibi.services.providers.utils.get_chibi_user", new=AsyncMock(return_value=user)),
        patch("chibi.services.providers.utils.get_available_skills", return_value=[]),
    ):
        prompt_json = await prepare_system_prompt("base", 1, interface, conversation_messages=conversation)

    prompt = json.loads(prompt_json)
    assert "tool_results_warning" in prompt
    warning = prompt["tool_results_warning"]
    assert f"web_search: ~{blob.estimate_tokens} tokens" in warning
    assert "clear_tool_call_history" in warning
    assert "SECRET_BLOB_CONTENT" not in warning


@pytest.mark.asyncio
async def test_prepare_system_prompt_tool_results_warning_ignores_shape_i_tool_messages() -> None:
    """In-loop shape-(i) role="tool" results must not trigger the volume-based warning."""
    _reset_usage_cache()
    user = _make_user()
    interface = cast(UserInterface, SimpleNamespace(thread_id=0, uses_uploaded_file_storage=False))

    conversation = [Message(role="tool", content="x" * 1200, tool_call_id="call_1")]

    store = UsageCacheStore()
    store._data[store._make_key(user_id=1, thread_id=0)] = 100

    with (
        patch("chibi.services.providers.utils.get_chibi_user", new=AsyncMock(return_value=user)),
        patch("chibi.services.providers.utils.get_available_skills", return_value=[]),
    ):
        prompt_json = await prepare_system_prompt("base", 1, interface, conversation_messages=conversation)

    prompt = json.loads(prompt_json)
    assert "tool_results_warning" not in prompt


@pytest.mark.asyncio
async def test_prepare_system_prompt_tool_results_warning_silent_below_share_threshold() -> None:
    """No volume-based warning while the shape-(ii) share stays at or below 25%."""
    _reset_usage_cache()
    user = _make_user()
    interface = cast(UserInterface, SimpleNamespace(thread_id=0, uses_uploaded_file_storage=False))

    conversation = [_make_tool_response_blob("web_search", "x" * 1200)]
    tool_results_tokens = sum(msg.estimate_tokens for msg in conversation)

    store = UsageCacheStore()
    store._data[store._make_key(user_id=1, thread_id=0)] = tool_results_tokens * 10

    with (
        patch("chibi.services.providers.utils.get_chibi_user", new=AsyncMock(return_value=user)),
        patch("chibi.services.providers.utils.get_available_skills", return_value=[]),
    ):
        prompt_json = await prepare_system_prompt("base", 1, interface, conversation_messages=conversation)

    prompt = json.loads(prompt_json)
    assert "tool_results_warning" not in prompt


@pytest.mark.asyncio
async def test_prepare_system_prompt_no_tool_results_warning_without_conversation() -> None:
    """A cached context size without retained conversation messages emits no volume-based warning."""
    _reset_usage_cache()
    user = _make_user()
    interface = cast(UserInterface, SimpleNamespace(thread_id=0, uses_uploaded_file_storage=False))

    store = UsageCacheStore()
    store._data[store._make_key(user_id=1, thread_id=0)] = 50000

    with (
        patch("chibi.services.providers.utils.get_chibi_user", new=AsyncMock(return_value=user)),
        patch("chibi.services.providers.utils.get_available_skills", return_value=[]),
    ):
        prompt_json = await prepare_system_prompt("base", 1, interface)

    prompt = json.loads(prompt_json)
    assert "tool_results_warning" not in prompt
