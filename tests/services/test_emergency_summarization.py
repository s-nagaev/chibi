"""Unit tests for emergency summarization (active model, max_tokens, input budget, cache invalidation)."""

import json
from copy import deepcopy
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from chibi.models import FunctionSchema, Message, ToolSchema
from chibi.schemas.app import ChatResponseSchema, UsageSchema
from chibi.services.usage_cache import UsageCacheStore
from chibi.services.user import (
    EMERGENCY_SUMMARIZATION_INPUT_BUDGET,
    _fit_messages_to_budget,
    _is_tool_response_message,
    emergency_summarization,
)


def _make_tool_response_blob(tool_name: str, filler: str) -> Message:
    """Build a shape-(ii) background tool-response turn (role="user" JSON blob).

    Args:
        tool_name: Name stored in the blob payload.
        filler: Payload content used to control the message size.

    Returns:
        The constructed message.
    """
    payload = {"type": "tool response", "tool_name": tool_name, "result": filler}
    return Message(role="user", content=json.dumps(payload))


def _make_shape_i_pair(tool_result_content: str) -> list[Message]:
    """Build a shape-(i) in-loop tool-call pair (assistant tool_calls + role="tool" result).

    Args:
        tool_result_content: Content for the role="tool" message.

    Returns:
        The assistant and tool messages forming the pair.
    """
    assistant = Message(
        role="assistant",
        content="Calling tool",
        tool_calls=[
            ToolSchema(id="call_1", type="function", function=FunctionSchema(name="read_file", arguments="{}"))
        ],
    )
    tool_result = Message(role="tool", content=tool_result_content, tool_call_id="call_1")
    return [assistant, tool_result]


def _make_summarization_mocks(messages: list[Message]) -> tuple[MagicMock, AsyncMock]:
    """Build a mock Database and a fake active provider for emergency_summarization.

    Args:
        messages: Conversation history returned by the mocked storage.

    Returns:
        A tuple of the database mock and the fake provider whose get_chat_response is an AsyncMock.
    """
    user = MagicMock()
    user.id = 1
    user.get_active_llm_model = Mock(return_value="active-model")
    provider = MagicMock()
    provider.get_chat_response = AsyncMock(
        return_value=(
            ChatResponseSchema(answer="summary", provider="test", model="active-model", usage=UsageSchema()),
            [],
        )
    )
    user.get_active_llm_provider = Mock(return_value=provider)

    db = MagicMock()
    db.get_or_create_user = AsyncMock(return_value=user)
    db.get_conversation_messages = AsyncMock(return_value=messages)
    db.add_message = AsyncMock()
    return db, provider


async def _run_emergency_summarization(db: MagicMock, **patches: Any) -> AsyncMock:
    """Run emergency_summarization with the storage reset stubbed out.

    Args:
        db: The mocked database to inject.
        **patches: Keyword arguments forwarded to ``patch`` over user-module gpt_settings attributes.

    Returns:
        The reset_chat_history AsyncMock used in place of the real reset.
    """
    with (
        patch("chibi.services.user.reset_chat_history", new=AsyncMock()) as reset_mock,
        patch("chibi.services.user.gpt_settings.max_tokens", 32000),
        patch("chibi.services.user.gpt_settings.max_history_tokens", patches.get("max_history_tokens", 100000)),
    ):
        await emergency_summarization.__wrapped__(db, storage_id=1, thread_id=0)
    return reset_mock


def _reset_usage_cache() -> None:
    """Clear the UsageCacheStore singleton between tests."""
    UsageCacheStore()._data.clear()


@pytest.mark.asyncio
async def test_emergency_summarization_passes_active_model_and_boosted_max_tokens() -> None:
    """The provider call must receive the thread's active model and int(max_tokens * 1.3)."""
    messages = [Message(role="user", content="hello"), Message(role="assistant", content="hi")]
    db, provider = _make_summarization_mocks(list(messages))

    await _run_emergency_summarization(db, max_history_tokens=100000)

    kwargs = provider.get_chat_response.await_args.kwargs
    assert kwargs["model"] == "active-model"
    assert kwargs["max_tokens"] == int(32000 * 1.3) == 41600
    assert kwargs["system_prompt"].startswith("Summarize this conversation")


@pytest.mark.asyncio
async def test_emergency_summarization_invalidates_usage_cache() -> None:
    """After a successful summarization the cached pre-reset prompt size must be dropped."""
    _reset_usage_cache()
    messages = [Message(role="user", content="hello"), Message(role="assistant", content="hi")]
    db, _provider = _make_summarization_mocks(list(messages))

    store = UsageCacheStore()
    store._data[store._make_key(user_id=1, thread_id=0)] = 99999

    await _run_emergency_summarization(db, max_history_tokens=100000)

    assert store.get(user_id=1, thread_id=0) is None


@pytest.mark.asyncio
async def test_shape_i_pairs_excluded_from_input_and_history_never_mutated() -> None:
    """Shape-(i) tool-call pairs never reach the summarizer input and stored history stays intact."""
    pair = _make_shape_i_pair(tool_result_content='{"status": "ok", "result": "SECRET_SHAPE_I_PAYLOAD"}')
    messages = [
        Message(role="user", content="real user turn"),
        *pair,
        Message(role="assistant", content="final answer"),
    ]
    snapshot = deepcopy(messages)
    db, provider = _make_summarization_mocks(messages)

    await _run_emergency_summarization(db, max_history_tokens=100000)

    sent_messages = provider.get_chat_response.await_args.kwargs["messages"]
    assert len(sent_messages) == 1
    sent_text = sent_messages[0].content
    assert "SECRET_SHAPE_I_PAYLOAD" not in sent_text
    assert "Calling tool" not in sent_text
    assert "real user turn" in sent_text
    assert "final answer" in sent_text
    assert messages == snapshot


@pytest.mark.asyncio
async def test_shape_ii_blobs_dropped_under_budget_and_recent_turn_kept() -> None:
    """Over-budget shape-(ii) blobs are dropped from the summarizer input, the latest turn is kept."""
    blob = _make_tool_response_blob("web_search", "BLOB_MARKER_PAYLOAD" + "x" * 400)
    messages = [
        Message(role="user", content="a" * 80),
        blob,
        Message(role="assistant", content="recent user turn"),
    ]
    snapshot = deepcopy(messages)
    db, provider = _make_summarization_mocks(messages)

    await _run_emergency_summarization(db, max_history_tokens=100)

    budget = int(100 * EMERGENCY_SUMMARIZATION_INPUT_BUDGET)
    total = sum(msg.estimate_tokens for msg in messages)
    assert budget < total
    assert budget >= total - blob.estimate_tokens
    sent_text = provider.get_chat_response.await_args.kwargs["messages"][0].content
    assert "BLOB_MARKER_PAYLOAD" not in sent_text
    assert "a" * 80 in sent_text
    assert "recent user turn" in sent_text
    assert messages == snapshot


def test_fit_messages_to_budget_drops_largest_blob_first() -> None:
    """With a budget reachable by dropping only the largest blob, the smaller blob must survive.

    Dropping smallest-first would have to remove both blobs to fit; keeping the small blob proves
    the largest-first ordering.
    """
    small_blob = _make_tool_response_blob("small_tool", "s" * 400)
    big_blob = _make_tool_response_blob("big_tool", "b" * 4000)
    old_turn = Message(role="user", content="old turn")
    recent = Message(role="assistant", content="most recent turn")
    messages = [old_turn, big_blob, small_blob, recent]

    total = sum(msg.estimate_tokens for msg in messages)
    budget = total - big_blob.estimate_tokens

    kept = _fit_messages_to_budget(deepcopy(messages), budget)

    assert [msg.content for msg in kept] == [old_turn.content, small_blob.content, recent.content]


def test_fit_messages_to_budget_drops_oldest_turns_after_blobs() -> None:
    """When blob-dropping is not enough, the oldest remaining turns go next."""
    blob = _make_tool_response_blob("tool", "x" * 400)
    first_turn = Message(role="user", content="first")
    second_turn = Message(role="assistant", content="second")
    recent = Message(role="assistant", content="most recent turn")
    messages = [first_turn, blob, second_turn, recent]

    total = sum(msg.estimate_tokens for msg in messages)
    budget = total - blob.estimate_tokens - first_turn.estimate_tokens

    kept = _fit_messages_to_budget(deepcopy(messages), budget)

    assert [msg.content for msg in kept] == [second_turn.content, recent.content]


def test_fit_messages_to_budget_never_drops_most_recent_message() -> None:
    """Even with a tiny budget, the most recent message must survive."""
    first = Message(role="user", content="x" * 400)
    recent = Message(role="assistant", content="most recent turn")
    messages = [first, recent]

    kept = _fit_messages_to_budget(deepcopy(messages), budget=1)

    assert [msg.content for msg in kept] == [recent.content]


def test_fit_messages_to_budget_noop_when_under_budget() -> None:
    """History already fitting the budget is returned unchanged."""
    messages = [Message(role="user", content="hello"), Message(role="assistant", content="hi")]
    total = sum(msg.estimate_tokens for msg in messages)

    kept = _fit_messages_to_budget(deepcopy(messages), budget=total)

    assert kept == messages


@pytest.mark.parametrize(
    ("content", "role", "expected"),
    [
        (json.dumps({"type": "tool response", "tool_name": "t", "result": "r"}), "user", True),
        (json.dumps({"type": "something else"}), "user", False),
        ("not json at all", "user", False),
        (json.dumps({"type": "tool response"}), "tool", False),
        ("", "user", False),
    ],
)
def test_is_tool_response_message(content: str, role: str, expected: bool) -> None:
    """Only role="user" JSON blobs with type "tool response" are background tool responses."""
    message = Message(role=role, content=content)

    assert _is_tool_response_message(message) is expected
