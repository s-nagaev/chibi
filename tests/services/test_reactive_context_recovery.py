"""Tests for reactive context-overflow recovery."""

from io import BytesIO
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from chibi.exceptions import ContextLengthExceededError, ServiceResponseError
from chibi.schemas.app import ChatResponseSchema, UsageSchema
from chibi.services.interface import UserInterface
from chibi.utils.app import handle_gpt_exceptions


class _FakeInterface(UserInterface):
    """Minimal fake interface for exercising the decorator."""

    def __init__(self, storage_id: int = 42, thread_id: int = 7) -> None:
        self._storage_id = storage_id
        self._thread_id = thread_id
        self.sent_messages: list[str] = []

    @property
    def chat_id(self) -> int:
        return self._storage_id

    @property
    def user_id(self) -> int:
        return self._storage_id

    @property
    def storage_id(self) -> int:
        return self._storage_id

    @property
    def thread_id(self) -> int:
        return self._thread_id

    @property
    def user_data(self) -> str:
        return f"User {self._storage_id}"

    @property
    def chat_data(self) -> str:
        return f"chat {self._storage_id}, thread {self._thread_id}"

    @property
    def attached_document(self) -> dict[str, str] | None:
        return None

    @property
    def attached_document_caption(self) -> str | None:
        return None

    async def get_text_prompt(self) -> str | None:
        return None

    async def get_voice_prompt(self) -> BytesIO | None:
        return None

    async def get_caption(self) -> str | None:
        return None

    def set_caption(self, caption: str) -> None:
        return None

    async def send_action_typing(self) -> None:
        return None

    async def send_action_uploading_photo(self) -> None:
        return None

    async def send_action_recording(self) -> None:
        return None

    async def send_reaction(self, reaction: str) -> None:
        return None

    async def delete_last_user_message(self) -> None:
        return None

    async def send_message(self, message: str, reply: bool = True, **kwargs: Any) -> None:
        self.sent_messages.append(message)

    async def send_audio(
        self,
        audio: bytes | str,
        reply: bool = True,
        title: str | None = None,
        caption: str | None = None,
        performer: str | None = None,
        duration: int | None = None,
        thumbnail: bytes | None = None,
        filename: str | None = None,
        **kwargs: Any,
    ) -> None:
        return None

    async def send_video(
        self,
        video: bytes | str,
        reply: bool = True,
        title: str | None = None,
        caption: str | None = None,
        duration: int | None = None,
        thumbnail: bytes | None = None,
        filename: str | None = None,
        **kwargs: Any,
    ) -> None:
        return None

    async def send_images(self, images: list[BytesIO] | list[str], reply: bool = True, **kwargs: Any) -> None:
        return None

    async def send_document(
        self,
        document: bytes | BytesIO,
        filename: str | None = None,
        caption: str | None = None,
        thumbnail: bytes | None = None,
        **kwargs: Any,
    ) -> None:
        return None

    async def create_thread(self, name: str) -> int:
        return 0

    async def rename_thread(self, new_name: str) -> bool:
        return True

    async def delete_thread(self) -> bool:
        return True


def _make_chat_response(answer: str = "ok") -> ChatResponseSchema:
    return ChatResponseSchema(
        answer=answer,
        provider="openai",
        model="gpt-4",
        usage=UsageSchema(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


@pytest.mark.asyncio
async def test_context_overflow_triggers_summarize_and_retry_once_then_success() -> None:
    """A single context_length_exceeded error is recovered by summarize + one retry."""
    call_count = 0

    @handle_gpt_exceptions
    async def handler(interface: UserInterface) -> ChatResponseSchema:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ContextLengthExceededError(provider="openai", model="gpt-4")
        return _make_chat_response("recovered")

    interface = _FakeInterface()

    with patch("chibi.services.user.emergency_summarization", new=AsyncMock()) as mock_summarize:
        result = await handler(interface=interface)

    assert call_count == 2
    mock_summarize.assert_awaited_once_with(storage_id=interface.storage_id, thread_id=interface.thread_id)
    assert isinstance(result, ChatResponseSchema)
    assert "recovered" in result.answer
    assert "compressed" in result.answer
    assert not interface.sent_messages


@pytest.mark.asyncio
async def test_context_overflow_twice_does_not_loop_and_apologizes() -> None:
    """A second overflow after summarization falls through to the apology path."""
    call_count = 0

    @handle_gpt_exceptions
    async def handler(interface: UserInterface) -> ChatResponseSchema:
        nonlocal call_count
        call_count += 1
        raise ContextLengthExceededError(provider="openai", model="gpt-4")

    interface = _FakeInterface()

    with patch("chibi.services.user.emergency_summarization", new=AsyncMock()) as mock_summarize:
        result = await handler(interface=interface)

    assert call_count == 2
    mock_summarize.assert_awaited_once_with(storage_id=interface.storage_id, thread_id=interface.thread_id)
    assert result is None
    assert len(interface.sent_messages) == 1
    assert "too large" in interface.sent_messages[0].lower()


@pytest.mark.asyncio
async def test_non_overflow_service_response_error_does_not_retry() -> None:
    """ServiceResponseError unrelated to context length logs and apologizes without retry."""
    call_count = 0

    @handle_gpt_exceptions
    async def handler(interface: UserInterface) -> ChatResponseSchema:
        nonlocal call_count
        call_count += 1
        raise ServiceResponseError(provider="openai", model="gpt-4")

    interface = _FakeInterface()

    with patch("chibi.services.user.emergency_summarization", new=AsyncMock()) as mock_summarize:
        result = await handler(interface=interface)

    assert call_count == 1
    mock_summarize.assert_not_awaited()
    assert result is None
    assert len(interface.sent_messages) == 1
    assert "unexpected response" in interface.sent_messages[0].lower()


@pytest.mark.asyncio
async def test_loop_guard_never_summarizes_more_than_once_per_turn() -> None:
    """Even if the retry path keeps raising, emergency_summarization is called at most once."""
    call_count = 0

    @handle_gpt_exceptions
    async def handler(interface: UserInterface) -> ChatResponseSchema:
        nonlocal call_count
        call_count += 1
        raise ContextLengthExceededError(provider="openai", model="gpt-4")

    interface = _FakeInterface()

    with patch("chibi.services.user.emergency_summarization", new=AsyncMock()) as mock_summarize:
        await handler(interface=interface)

    mock_summarize.assert_awaited_once()


@pytest.mark.asyncio
async def test_reactive_recovery_disabled_falls_through_to_apology() -> None:
    """When reactive_context_recovery is False, overflow is treated as a fatal context error."""

    @handle_gpt_exceptions
    async def handler(interface: UserInterface) -> ChatResponseSchema:
        raise ContextLengthExceededError(provider="openai", model="gpt-4")

    interface = _FakeInterface()

    with (
        patch("chibi.config.gpt_settings.reactive_context_recovery", False),
        patch("chibi.services.user.emergency_summarization", new=AsyncMock()) as mock_summarize,
    ):
        result = await handler(interface=interface)

    mock_summarize.assert_not_awaited()
    assert result is None
    assert len(interface.sent_messages) == 1
    assert "too large" in interface.sent_messages[0].lower()


@pytest.mark.asyncio
async def test_recovery_appends_brief_note_only_to_chat_response() -> None:
    """The compressed-context note is added only when the retry returns a ChatResponseSchema."""
    call_count = 0

    @handle_gpt_exceptions
    async def handler(interface: UserInterface) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ContextLengthExceededError(provider="openai", model="gpt-4")
        return "plain string result"

    interface = _FakeInterface()

    with patch("chibi.services.user.emergency_summarization", new=AsyncMock()) as mock_summarize:
        result = await handler(interface=interface)

    mock_summarize.assert_awaited_once()
    assert result == "plain string result"
    assert not interface.sent_messages


@pytest.mark.asyncio
async def test_recovery_success_even_when_usage_cache_store_is_empty() -> None:
    """Reactive recovery does not depend on the proactive usage-cache threshold."""
    call_count = 0

    @handle_gpt_exceptions
    async def handler(interface: UserInterface) -> ChatResponseSchema:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ContextLengthExceededError(provider="openai", model="gpt-4")
        return _make_chat_response("recovered")

    interface = _FakeInterface()

    with (
        patch("chibi.services.usage_cache.UsageCacheStore.get", return_value=None),
        patch("chibi.services.user.emergency_summarization", new=AsyncMock()) as mock_summarize,
    ):
        result = await handler(interface=interface)

    assert call_count == 2
    mock_summarize.assert_awaited_once()
    assert isinstance(result, ChatResponseSchema)
