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
    collect_volatile_context_note,
    get_usage_from_anthropic_response,
    is_context_overflow_signal,
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
async def test_prepare_system_prompt_omits_volatile_keys() -> None:
    """The volatile keys never travel in the system payload, even above the warning threshold.

    They moved to the per-turn context note attached to the last converted user
    message by the provider entry points (collect_volatile_context_note).
    """
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
    assert "approximate_context_size" not in prompt
    assert "context_size_warning" not in prompt
    assert "tool_results_warning" not in prompt


def test_collect_volatile_context_note_empty_without_interface() -> None:
    """Sub-agent and other interface-less requests carry no note, as before the volatile keys."""
    _reset_usage_cache()
    assert collect_volatile_context_note(user_id=1, interface=None) == ""


def test_collect_volatile_context_note_na_when_store_empty() -> None:
    """Empty store → the note reports 'n/a' and carries no warning."""
    _reset_usage_cache()
    interface = cast(UserInterface, SimpleNamespace(thread_id=1, uses_uploaded_file_storage=False))
    conversation = [
        Message(role="user", content="Hello there"),
        Message(role="assistant", content="General Kenobi"),
    ]

    note = collect_volatile_context_note(user_id=1, interface=interface, conversation_messages=conversation)

    assert "approximate_context_size: n/a" in note
    assert "context_size_warning" not in note


def test_collect_volatile_context_note_shows_real_value_and_percentage() -> None:
    """Non-empty store → the note shows the real token count plus percentage of max_history_tokens."""
    _reset_usage_cache()
    interface = cast(UserInterface, SimpleNamespace(thread_id=1, uses_uploaded_file_storage=False))

    store = UsageCacheStore()
    store._data[store._make_key(user_id=1, thread_id=1)] = 58372

    with (
        patch("chibi.config.gpt.gpt_settings.max_history_tokens", 200000),
        patch("chibi.config.gpt.gpt_settings.context_size_warning_threshold", 50),
    ):
        note = collect_volatile_context_note(user_id=1, interface=interface)

    expected_pct = round(58372 / 200000 * 100)
    assert f"approximate_context_size: 58,372 tokens ({expected_pct}% of {200000:,} limit)" in note
    assert "context_size_warning" not in note


def test_collect_volatile_context_note_warning_fires_above_threshold() -> None:
    """Warning is emitted when the real context exceeds the configured threshold percentage."""
    _reset_usage_cache()
    interface = cast(UserInterface, SimpleNamespace(thread_id=0, uses_uploaded_file_storage=False))

    store = UsageCacheStore()
    store._data[store._make_key(user_id=1, thread_id=0)] = 120000

    with patch("chibi.config.gpt.gpt_settings.context_size_warning_threshold", 50):
        note = collect_volatile_context_note(user_id=1, interface=interface)

    assert "context_size_warning" in note
    assert "50%" in note


@pytest.mark.parametrize("stored_tokens", [50000, 40000])
def test_collect_volatile_context_note_warning_silent_at_or_below_threshold(stored_tokens: int) -> None:
    """Warning is NOT emitted when the real context is at or below the threshold.

    50000 tokens equals exactly 50% of the pinned limit (boundary of the strict ``>`` check),
    40000 tokens is clearly below it.
    """
    _reset_usage_cache()
    interface = cast(UserInterface, SimpleNamespace(thread_id=0, uses_uploaded_file_storage=False))

    store = UsageCacheStore()
    store._data[store._make_key(user_id=1, thread_id=0)] = stored_tokens

    with (
        patch("chibi.config.gpt.gpt_settings.max_history_tokens", 100000),
        patch("chibi.config.gpt.gpt_settings.context_size_warning_threshold", 50),
    ):
        note = collect_volatile_context_note(user_id=1, interface=interface)

    expected_pct = round(stored_tokens / 100000 * 100)
    assert expected_pct <= 50
    assert "context_size_warning" not in note


def test_collect_volatile_context_note_warning_threshold_from_config() -> None:
    """The warning threshold is read from config, not hardcoded."""
    _reset_usage_cache()
    interface = cast(UserInterface, SimpleNamespace(thread_id=0, uses_uploaded_file_storage=False))

    store = UsageCacheStore()
    store._data[store._make_key(user_id=1, thread_id=0)] = 60000

    with patch("chibi.config.gpt.gpt_settings.context_size_warning_threshold", 25):
        note = collect_volatile_context_note(user_id=1, interface=interface)

    assert "context_size_warning" in note
    assert "25%" in note


def test_collect_volatile_context_note_uses_correct_key_matching_write_side() -> None:
    """The read key (user_id, thread_id) matches the write-side convention."""
    _reset_usage_cache()
    interface = cast(UserInterface, SimpleNamespace(thread_id=3, uses_uploaded_file_storage=False))

    store = UsageCacheStore()
    store.store(user_id=1, thread_id=3, usage=UsageSchema(prompt_tokens=12345), provider="openai")

    note = collect_volatile_context_note(user_id=1, interface=interface)

    assert "12,345 tokens" in note


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


def test_collect_volatile_context_note_warning_is_informative_not_imperative() -> None:
    """The context warning reports the size and names the tools, and commands nothing.

    Compaction is decided and performed by the pre-call budget guard before the
    request is sent, and the compaction policy lives in the base system prompt;
    the volatile warning stays a statement of state and keeps the
    '{threshold}%' substring.
    """
    _reset_usage_cache()
    interface = cast(UserInterface, SimpleNamespace(thread_id=0, uses_uploaded_file_storage=False))

    store = UsageCacheStore()
    store._data[store._make_key(user_id=1, thread_id=0)] = 120000

    with patch("chibi.config.gpt.gpt_settings.context_size_warning_threshold", 50):
        note = collect_volatile_context_note(user_id=1, interface=interface)

    warning_start = note.index("context_size_warning")
    warning = note[warning_start:]
    lowered = warning.lower()
    assert "mandatory" not in lowered
    assert "must" not in lowered
    assert "before producing" not in lowered
    assert "do not ignore" not in lowered
    assert "continue answering" not in lowered
    assert "summarize_history" in warning
    assert "clear_tool_call_history" in warning
    assert "50%" in warning


def test_collect_volatile_context_note_is_marker_delimited() -> None:
    """The note is a delimited block so attach and strip can detect it on any payload shape."""
    _reset_usage_cache()
    interface = cast(UserInterface, SimpleNamespace(thread_id=1, uses_uploaded_file_storage=False))

    store = UsageCacheStore()
    store._data[store._make_key(user_id=1, thread_id=1)] = 100

    note = collect_volatile_context_note(user_id=1, interface=interface)

    assert note.startswith("<context_note>\n")
    assert note.endswith("\n</context_note>")


def test_collect_volatile_context_note_tool_results_warning_fires_above_share_threshold() -> None:
    """The volume-based warning fires when shape-(ii) blobs exceed 25% of the real context size."""
    _reset_usage_cache()
    interface = cast(UserInterface, SimpleNamespace(thread_id=0, uses_uploaded_file_storage=False))

    blob = _make_tool_response_blob("web_search", "SECRET_BLOB_CONTENT" + "x" * 1200)
    conversation = [blob, Message(role="assistant", content="ok")]
    tool_results_tokens = sum(msg.estimate_tokens for msg in conversation if _is_tool_response_message(msg))

    store = UsageCacheStore()
    store._data[store._make_key(user_id=1, thread_id=0)] = tool_results_tokens * 3

    note = collect_volatile_context_note(user_id=1, interface=interface, conversation_messages=conversation)

    assert "tool_results_warning" in note
    assert f"web_search: ~{blob.estimate_tokens} tokens" in note
    assert "clear_tool_call_history" in note
    assert "you should call" not in note.lower()
    assert "SECRET_BLOB_CONTENT" not in note


def test_collect_volatile_context_note_tool_results_warning_ignores_shape_i_tool_messages() -> None:
    """In-loop shape-(i) role="tool" results must not trigger the volume-based warning."""
    _reset_usage_cache()
    interface = cast(UserInterface, SimpleNamespace(thread_id=0, uses_uploaded_file_storage=False))

    conversation = [Message(role="tool", content="x" * 1200, tool_call_id="call_1")]

    store = UsageCacheStore()
    store._data[store._make_key(user_id=1, thread_id=0)] = 100

    note = collect_volatile_context_note(user_id=1, interface=interface, conversation_messages=conversation)

    assert "tool_results_warning" not in note


def test_collect_volatile_context_note_tool_results_warning_silent_below_share_threshold() -> None:
    """No volume-based warning while the shape-(ii) share stays at or below 25%."""
    _reset_usage_cache()
    interface = cast(UserInterface, SimpleNamespace(thread_id=0, uses_uploaded_file_storage=False))

    conversation = [_make_tool_response_blob("web_search", "x" * 1200)]
    tool_results_tokens = sum(msg.estimate_tokens for msg in conversation)

    store = UsageCacheStore()
    store._data[store._make_key(user_id=1, thread_id=0)] = tool_results_tokens * 10

    note = collect_volatile_context_note(user_id=1, interface=interface, conversation_messages=conversation)

    assert "tool_results_warning" not in note


def test_collect_volatile_context_note_no_tool_results_warning_without_conversation() -> None:
    """A cached context size without retained conversation messages emits no volume-based warning."""
    _reset_usage_cache()
    interface = cast(UserInterface, SimpleNamespace(thread_id=0, uses_uploaded_file_storage=False))

    store = UsageCacheStore()
    store._data[store._make_key(user_id=1, thread_id=0)] = 50000

    note = collect_volatile_context_note(user_id=1, interface=interface)

    assert "tool_results_warning" not in note


@pytest.mark.parametrize(
    ("status_code", "detail"),
    [
        (400, {"error": {"message": "test", "type": "invalid_request_error", "code": "context_length_exceeded"}}),
        (400, {"error": {"code": 400, "message": "This model's maximum context length is 8192 tokens."}}),
        (
            400,
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "prompt is too long: 250000 tokens > 200000 maximum",
                },
            },
        ),
        (
            400,
            {
                "error": {
                    "code": 400,
                    "message": "The input token count (12345) exceeds the maximum number of tokens allowed (1000).",
                    "status": "INVALID_ARGUMENT",
                }
            },
        ),
        (400, 'Status 400. Body: {"object": "error", "message": "Too many tokens in request"}'),
        (413, {"error": {"message": "Request too large: too many tokens in the request"}}),
    ],
)
def test_is_context_overflow_signal_matches_provider_wordings(status_code: int, detail: Any) -> None:
    """Every known provider wording of an overflow is recognized."""
    assert is_context_overflow_signal(status_code=status_code, detail=detail)


@pytest.mark.parametrize(
    ("status_code", "detail"),
    [
        (429, {"error": {"message": "prompt is too long: too many requests"}}),
        (500, {"error": {"message": "exceeds the maximum number of tokens"}}),
        (400, {"error": {"message": "Invalid image data", "type": "invalid_request_error"}}),
        (400, None),
        # Loose phrases that are not overflow-specific: an unrelated 400 whose
        # body merely contains them must keep its original error type.
        (400, {"error": {"message": "Request too large", "type": "invalid_request_error"}}),
        (400, {"error": {"message": "Please reduce the length of your input"}}),
        (400, {"error": {"message": "The input token count is missing", "status": "INVALID_ARGUMENT"}}),
        (400, {"error": {"message": "The input token count (1) does not match the message"}}),
    ],
)
def test_is_context_overflow_signal_rejects_non_overflow_errors(status_code: int, detail: Any) -> None:
    """Non-overflow errors (wrong status, unrelated message, missing payload) are not flagged."""
    assert not is_context_overflow_signal(status_code=status_code, detail=detail)
