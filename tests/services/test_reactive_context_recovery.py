"""Tests for the reactive context-overflow recovery at the budget choke point.

The recovery lives at the single provider call site
(``chibi.services.user.get_llm_chat_completion_answer`` ->
``_reactive_context_recovery``); ``handle_gpt_exceptions`` is a pure error
presenter and owns no recovery logic.
"""

import asyncio
from contextlib import contextmanager
from io import BytesIO
from typing import Any, Iterator
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from loguru import logger

from chibi.exceptions import ContextLengthExceededError, ServiceResponseError
from chibi.schemas.app import ChatResponseSchema, UsageSchema
from chibi.services.context_measurement import SizeProvenance
from chibi.services.interface import UserInterface
from chibi.services.lock_manager import LockManager
from chibi.services.user import CompactionOutcome, CompactionResult, _reactive_context_recovery
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


def _make_chat_response(answer: str = "ok", prompt_tokens: int = 1) -> ChatResponseSchema:
    return ChatResponseSchema(
        answer=answer,
        provider="openai",
        model="gpt-4",
        usage=UsageSchema(prompt_tokens=prompt_tokens, completion_tokens=1, total_tokens=prompt_tokens + 1),
    )


@contextmanager
def _captured_logs(min_level: str) -> Iterator[list[str]]:
    """Capture loguru output into a list for assertions."""
    lines: list[str] = []
    sink_id = logger.add(lines.append, level=min_level)
    try:
        yield lines
    finally:
        logger.remove(sink_id)


@pytest.mark.asyncio
async def test_overflow_is_presented_without_any_recovery() -> None:
    """handle_gpt_exceptions presents the overflow error: no retry, no summarization."""
    call_count = 0

    @handle_gpt_exceptions
    async def handler(interface: UserInterface) -> ChatResponseSchema:
        nonlocal call_count
        call_count += 1
        raise ContextLengthExceededError(provider="openai", model="gpt-4")

    interface = _FakeInterface()

    with patch("chibi.services.user.emergency_summarization", new=AsyncMock()) as mock_summarize:
        result = await handler(interface=interface)

    assert call_count == 1
    mock_summarize.assert_not_awaited()
    assert result is None
    assert len(interface.sent_messages) == 1
    assert "too large" in interface.sent_messages[0].lower()


@pytest.mark.asyncio
async def test_non_overflow_service_response_error_is_presented_without_retry() -> None:
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


def test_context_overflow_is_never_a_failover_trigger() -> None:
    """The normalized overflow error stays excluded from failover on every path.

    ``classify_exception`` matches on exception type only, so it never consults
    the provider label and one assertion covers all five paths. The real
    per-path difference is the error type each path used to raise before
    normalization: Gemini's ``ServiceResponseError`` is a failover trigger,
    which is exactly what the typed normalization removes.
    """
    from chibi.services.failover import classify_exception

    assert classify_exception(ContextLengthExceededError(provider="openai", model="m")) is None
    assert classify_exception(ContextLengthExceededError(provider="gemini", model="m")) is None
    assert classify_exception(ContextLengthExceededError(provider="mistral", model="m")) is None
    assert classify_exception(ContextLengthExceededError(provider="anthropic", model="m")) is None
    assert classify_exception(ContextLengthExceededError(provider="minimax", model="m")) is None
    assert classify_exception(ServiceResponseError(provider="gemini", model="m")) == "server_error"


def _make_recovery_mocks(history: list[Any], prompt_tokens: int = 150) -> tuple[MagicMock, AsyncMock]:
    """Build a db double and a provider double for recovery tests."""
    user = MagicMock()
    user.id = 1
    provider = AsyncMock()
    provider.name = "openai"
    provider._get_max_tokens_value = Mock(return_value=32000)
    provider.get_chat_response = AsyncMock(
        return_value=(
            _make_chat_response("recovered", prompt_tokens=prompt_tokens),
            [MagicMock()],
        )
    )
    user.get_active_llm_provider = Mock(return_value=provider)

    db = MagicMock()
    db.get_or_create_user = AsyncMock(return_value=user)
    db.get_conversation_messages = AsyncMock(return_value=list(history))
    db.add_message = AsyncMock()
    return db, provider


async def _run_recovery(
    db: MagicMock,
    provider: AsyncMock,
    pre_tokens: int,
    post_tokens: int,
    compaction: Any = None,
) -> tuple[Any, AsyncMock]:
    """Run _reactive_context_recovery with the measurement entry point stubbed."""
    turn_message = MagicMock()
    overflow = ContextLengthExceededError(provider="openai", model="gpt-4")
    with (
        patch(
            "chibi.services.user.measure_request_size_with_provenance",
            new=AsyncMock(
                side_effect=[(pre_tokens, SizeProvenance.ESTIMATED), (post_tokens, SizeProvenance.ESTIMATED)]
            ),
        ),
        patch(
            "chibi.services.user.emergency_summarization",
            new=AsyncMock(return_value=compaction),
        ) as mock_compaction,
        patch("chibi.services.user.get_model_context_window", return_value=None),
        patch("chibi.services.user.gpt_settings.max_tokens", 32000),
    ):
        result = await _reactive_context_recovery.__wrapped__(
            db,
            storage_id=1,
            thread_id=0,
            user=MagicMock(),
            interface=MagicMock(),
            provider=provider,
            model="gpt-4",
            new_message_to_llm=turn_message,
            overflow_error=overflow,
        )
    return result, mock_compaction


@pytest.mark.asyncio
async def test_recovery_retries_once_with_reassembled_request_and_verifies_authoritatively() -> None:
    """A reduced request is retried once; the provider-reported usage is verified authoritatively."""
    history = [MagicMock(role="user", content="old"), MagicMock(role="assistant", content="newer")]
    db, provider = _make_recovery_mocks(history)
    turn_message = MagicMock()

    with _captured_logs("INFO") as logs:
        with (
            patch(
                "chibi.services.user.measure_request_size_with_provenance",
                new=AsyncMock(side_effect=[(1000, SizeProvenance.ESTIMATED), (100, SizeProvenance.ESTIMATED)]),
            ),
            patch(
                "chibi.services.user.emergency_summarization",
                new=AsyncMock(return_value=CompactionResult(outcome=CompactionOutcome.PREDICTED_FITS)),
            ) as mock_compaction,
            patch("chibi.services.user.get_model_context_window", return_value=None),
            patch("chibi.services.user.gpt_settings.max_tokens", 32000),
        ):
            result = await _reactive_context_recovery.__wrapped__(
                db,
                storage_id=1,
                thread_id=0,
                user=MagicMock(),
                interface=MagicMock(),
                provider=provider,
                model="gpt-4",
                new_message_to_llm=turn_message,
                overflow_error=ContextLengthExceededError(provider="openai", model="gpt-4"),
            )

    mock_compaction.assert_awaited_once_with(storage_id=1, thread_id=0)
    provider.get_chat_response.assert_awaited_once()
    sent = provider.get_chat_response.await_args.kwargs["messages"]
    # The retry is re-assembled from storage (the compacted history) plus the
    # current turn's message, never from the failed request's list object.
    assert len(sent) == len(history) + 1
    assert sent[-1] is turn_message
    assert result.answer.endswith("_(Context was compressed to stay within limits.)_")
    # The turn message and the answer are persisted.
    assert db.add_message.await_count == 2
    assert any("Authoritative compaction verification" in line for line in logs)
    assert any("provider reported 150 prompt tokens" in line for line in logs)


@pytest.mark.asyncio
async def test_recovery_fails_honestly_when_the_request_was_not_reduced() -> None:
    """A skipped compaction (None) leaves the request the same size: no retry, honest failure."""
    db, provider = _make_recovery_mocks([MagicMock(role="user", content="old")])

    with pytest.raises(ContextLengthExceededError):
        await _run_recovery(db, provider, pre_tokens=1000, post_tokens=1000, compaction=None)

    provider.get_chat_response.assert_not_awaited()
    db.add_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_recovery_skips_authoritative_verification_without_reported_usage() -> None:
    """No usage on the retry call: the verification is skipped, the advisory outcome stands."""
    db, provider = _make_recovery_mocks([MagicMock(role="user", content="old")])
    provider.get_chat_response = AsyncMock(
        return_value=(
            ChatResponseSchema(answer="recovered", provider="openai", model="gpt-4", usage=None),
            [],
        )
    )

    with _captured_logs("WARNING") as logs:
        result, _mock = await _run_recovery(
            db,
            provider,
            pre_tokens=1000,
            post_tokens=100,
            compaction=CompactionResult(outcome=CompactionOutcome.PREDICTED_FITS),
        )

    assert any("Authoritative compaction verification for user 1 thread 0 skipped" in line for line in logs)
    assert result.answer.endswith("_(Context was compressed to stay within limits.)_")


@pytest.mark.asyncio
async def test_recovery_disabled_raises_the_overflow_error() -> None:
    """reactive_context_recovery=False: the recovery is skipped and the error propagates."""
    db, provider = _make_recovery_mocks([MagicMock(role="user", content="old")])

    with (
        patch("chibi.services.user.gpt_settings.reactive_context_recovery", False),
        patch("chibi.services.user.emergency_summarization", new=AsyncMock()) as mock_compaction,
        patch(
            "chibi.services.user.measure_request_size_with_provenance",
            new=AsyncMock(return_value=(1000, SizeProvenance.ESTIMATED)),
        ),
    ):
        with pytest.raises(ContextLengthExceededError):
            await _reactive_context_recovery.__wrapped__(
                db,
                storage_id=1,
                thread_id=0,
                user=MagicMock(),
                interface=MagicMock(),
                provider=provider,
                model="gpt-4",
                new_message_to_llm=MagicMock(),
                overflow_error=ContextLengthExceededError(provider="openai", model="gpt-4"),
            )

    mock_compaction.assert_not_awaited()
    provider.get_chat_response.assert_not_awaited()


def _stub_user(provider: AsyncMock, model: str) -> Mock:
    """A minimal user double with an empty provider registry."""
    user = Mock()
    user.id = 1
    user.providers = Mock()
    user.providers.chat_ready = {}
    user.stt_provider = None
    user.get_active_llm_provider = Mock(return_value=provider)
    user.get_active_llm_model = Mock(return_value=model)
    return user


def _stub_interface() -> Mock:
    """A minimal interface double for the full-path tests."""
    interface = Mock()
    interface.storage_id = 1
    interface.thread_id = 0
    interface.user_data = {"first_name": "Tester"}
    interface.editor_context = None
    return interface


@pytest.mark.asyncio
async def test_full_path_recovery_runs_outside_the_thread_lock(monkeypatch) -> None:
    """The overflow recovery re-enters the per-thread lock only after the caller released it.

    The provider raises overflow on the first attempt (inside the caller's
    ``async with lock``); the fake compaction observes the lock state at the
    moment it runs. If a regression ever moved the recovery inside the held
    lock, the fake would see the lock taken and this test would fail.
    """
    from chibi.services.user import get_llm_chat_completion_answer
    from chibi.storage.database import _db_provider

    lock_state: dict[str, bool] = {}

    async def fake_summarization(storage_id: int, thread_id: int) -> None:
        lock = await LockManager().get_lock(key=f"{storage_id}_{thread_id}")
        lock_state["locked"] = lock.locked()

    provider = AsyncMock()
    provider.name = "openai"
    provider._get_max_tokens_value = Mock(return_value=32000)
    provider.get_chat_response = AsyncMock(
        side_effect=[
            ContextLengthExceededError(provider="openai", model="gpt-5.6-luna"),
            (
                _make_chat_response("recovered", prompt_tokens=150),
                [MagicMock()],
            ),
        ]
    )
    user = _stub_user(provider, "gpt-5.6-luna")

    db = AsyncMock()
    db.get_or_create_user = AsyncMock(return_value=user)
    db.get_conversation_messages = AsyncMock(return_value=[MagicMock(role="user", content="old")])
    db.add_message = AsyncMock()

    monkeypatch.setattr(_db_provider, "get_database", AsyncMock(return_value=db))
    with (
        patch("chibi.services.user.emergency_summarization", new=fake_summarization),
        patch(
            "chibi.services.user.measure_request_size_with_provenance",
            new=AsyncMock(side_effect=[(1000, SizeProvenance.ESTIMATED), (100, SizeProvenance.ESTIMATED)]),
        ),
        patch("chibi.services.user.get_model_context_window", return_value=None),
    ):
        result = await get_llm_chat_completion_answer(storage_id=1, interface=_stub_interface(), user_text_message="hi")

    assert lock_state["locked"] is False
    assert provider.get_chat_response.await_count == 2
    assert result.answer.endswith("_(Context was compressed to stay within limits.)_")
    # The turn message and the answer are persisted once.
    assert db.add_message.await_count == 2


@pytest.mark.asyncio
async def test_full_path_overflow_after_failed_recovery_is_presented(monkeypatch) -> None:
    """When the recovery cannot reduce the request, the caller re-raises the original error."""
    from chibi.services.user import get_llm_chat_completion_answer
    from chibi.storage.database import _db_provider

    provider = AsyncMock()
    provider.name = "openai"
    provider._get_max_tokens_value = Mock(return_value=32000)
    provider.get_chat_response = AsyncMock(
        side_effect=ContextLengthExceededError(provider="openai", model="gpt-5.6-luna")
    )
    user = _stub_user(provider, "gpt-5.6-luna")

    db = AsyncMock()
    db.get_or_create_user = AsyncMock(return_value=user)
    db.get_conversation_messages = AsyncMock(return_value=[MagicMock(role="user", content="old")])
    db.add_message = AsyncMock()

    monkeypatch.setattr(_db_provider, "get_database", AsyncMock(return_value=db))
    with (
        patch("chibi.services.user.emergency_summarization", new=AsyncMock(return_value=None)),
        patch(
            "chibi.services.user.measure_request_size_with_provenance",
            new=AsyncMock(side_effect=[(1000, SizeProvenance.ESTIMATED), (1000, SizeProvenance.ESTIMATED)]),
        ),
        patch("chibi.services.user.get_model_context_window", return_value=None),
    ):
        with pytest.raises(ContextLengthExceededError):
            await get_llm_chat_completion_answer(storage_id=1, interface=_stub_interface(), user_text_message="hi")

    # The doomed retry never happened: the provider was called exactly once.
    assert provider.get_chat_response.await_count == 1


@pytest.mark.asyncio
async def test_full_path_normal_turn_still_persists_and_returns(monkeypatch) -> None:
    """A normal (non-overflow) turn keeps its original persistence tail."""
    from chibi.services.user import get_llm_chat_completion_answer
    from chibi.storage.database import _db_provider

    provider = AsyncMock()
    provider.name = "openai"
    provider._get_max_tokens_value = Mock(return_value=32000)
    provider.get_chat_response = AsyncMock(
        return_value=(
            _make_chat_response("primary answer", prompt_tokens=10),
            [MagicMock(role="assistant", content="primary answer")],
        )
    )
    user = _stub_user(provider, "gpt-5.6-luna")

    db = AsyncMock()
    db.get_or_create_user = AsyncMock(return_value=user)
    db.get_conversation_messages = AsyncMock(return_value=[MagicMock(role="user", content="old")])
    db.add_message = AsyncMock()

    monkeypatch.setattr(_db_provider, "get_database", AsyncMock(return_value=db))
    with (
        patch(
            "chibi.services.user.measure_request_size_with_provenance",
            new=AsyncMock(return_value=(100, SizeProvenance.ESTIMATED)),
        ),
        patch("chibi.services.user.get_model_context_window", return_value=None),
    ):
        result = await asyncio.wait_for(
            get_llm_chat_completion_answer(storage_id=1, interface=_stub_interface(), user_text_message="hi"),
            timeout=5,
        )

    assert result.answer == "primary answer"
    # The turn message and the assistant answer are persisted.
    assert db.add_message.await_count == 2
