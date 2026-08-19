"""Tests for IDE editor-context injection into LLM prompts."""

import json
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

import chibi.config  # noqa: F401
from chibi.runners.ide_transport import IDEInterface
from chibi.schemas.app import ChatResponseSchema, UsageSchema
from chibi.services.interface import UserInterface
from chibi.services.user import get_llm_chat_completion_answer


class PlainInterface:
    """Minimal non-IDE interface carrying a Telegram-like context object."""

    storage_id = 1
    thread_id = 0
    user_data = "Telegram user"
    context = object()


@pytest.fixture
def fake_user() -> SimpleNamespace:
    """Provide a user with a recording LLM provider."""
    provider = AsyncMock()
    provider.get_chat_response.return_value = (
        ChatResponseSchema(answer="ok", provider="test-provider", model="test-model", usage=UsageSchema()),
        [],
    )
    return SimpleNamespace(
        stt_provider=None,
        get_active_llm_provider=lambda thread_id: provider,
        get_active_llm_model=lambda thread_id: "test-model",
        provider=provider,
    )


@pytest.mark.asyncio
async def test_non_ide_prompt_has_unchanged_shape(fake_user: SimpleNamespace) -> None:
    """A Telegram-like interface must not leak its context into the prompt."""
    db = AsyncMock()
    db.get_or_create_user.return_value = fake_user
    db.get_conversation_messages.return_value = []

    await get_llm_chat_completion_answer.__wrapped__(
        db, 1, cast(UserInterface, PlainInterface()), user_text_message="hello"
    )

    message = fake_user.provider.get_chat_response.await_args.kwargs["messages"][-1]
    prompt = json.loads(message.content)
    assert set(prompt) == {"user", "prompt", "datetime_now", "type", "transcribed_from_voice_message"}
    assert "editor_context" not in prompt


@pytest.mark.asyncio
async def test_ide_prompt_contains_sanitized_editor_context(fake_user: SimpleNamespace) -> None:
    """An IDE request includes canonical, bounded editor context."""
    db = AsyncMock()
    db.get_or_create_user.return_value = fake_user
    db.get_conversation_messages.return_value = []
    interface = IDEInterface(
        7,
        "explain",
        {
            "active_file": "foo.py",
            "selection": {"start_line": 1, "end_line": 2, "text": "bar"},
            "language_id": "python",
            "workspace_root": "/private/repository",
            "cursor_position": {"line": 2, "character": 3},
        },
        lambda _: None,
    )

    await get_llm_chat_completion_answer.__wrapped__(db, 1, interface, user_text_message="explain")

    message = fake_user.provider.get_chat_response.await_args.kwargs["messages"][-1]
    context = json.loads(message.content)["editor_context"]
    assert context == {
        "active_file": "foo.py",
        "selection": {"start_line": 1, "end_line": 2, "text": "bar"},
        "language_id": "python",
        "workspace_root": "repository",
        "cursor_position": {"line": 2, "character": 3},
    }


@pytest.mark.asyncio
async def test_empty_editor_context_omits_section(fake_user: SimpleNamespace) -> None:
    """An IDE request with an empty editor_context dict must not add the section."""
    db = AsyncMock()
    db.get_or_create_user.return_value = fake_user
    db.get_conversation_messages.return_value = []
    interface = IDEInterface(7, "hello", {}, lambda _: None)

    await get_llm_chat_completion_answer.__wrapped__(db, 1, interface, user_text_message="hello")

    message = fake_user.provider.get_chat_response.await_args.kwargs["messages"][-1]
    prompt = json.loads(message.content)
    assert "editor_context" not in prompt


@pytest.mark.asyncio
async def test_selection_text_truncated_to_4kb(fake_user: SimpleNamespace) -> None:
    """Long selection text is truncated to 4096 bytes."""
    db = AsyncMock()
    db.get_or_create_user.return_value = fake_user
    db.get_conversation_messages.return_value = []
    long_text = "x" * 5000
    interface = IDEInterface(
        7,
        "refactor",
        {
            "active_file": "big.py",
            "selection": {"start_line": 10, "end_line": 20, "text": long_text},
            "language_id": "python",
            "workspace_root": "/home/user/project",
            "cursor_position": {"line": 15, "character": 0},
        },
        lambda _: None,
    )

    await get_llm_chat_completion_answer.__wrapped__(db, 1, interface, user_text_message="refactor")

    message = fake_user.provider.get_chat_response.await_args.kwargs["messages"][-1]
    context = json.loads(message.content)["editor_context"]
    assert context["active_file"] == "big.py"
    assert context["language_id"] == "python"
    assert context["workspace_root"] == "project"
    assert context["selection"]["text"] == long_text[:4096]
    assert len(context["selection"]["text"]) == 4096


@pytest.mark.asyncio
async def test_workspace_root_basename_sanitized(fake_user: SimpleNamespace) -> None:
    """Absolute workspace_root is reduced to its basename."""
    db = AsyncMock()
    db.get_or_create_user.return_value = fake_user
    db.get_conversation_messages.return_value = []
    interface = IDEInterface(
        7,
        "explain",
        {
            "active_file": "foo.py",
            "selection": None,
            "language_id": "python",
            "workspace_root": "/Users/sergio/Develop/personal/chibi",
            "cursor_position": {"line": 1, "character": 1},
        },
        lambda _: None,
    )

    await get_llm_chat_completion_answer.__wrapped__(db, 1, interface, user_text_message="explain")

    message = fake_user.provider.get_chat_response.await_args.kwargs["messages"][-1]
    context = json.loads(message.content)["editor_context"]
    assert context["workspace_root"] == "chibi"
    assert "/Users/sergio" not in json.dumps(context)
