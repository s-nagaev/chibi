"""Integration tests for the scheduler self-wake infrastructure."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram.constants import ParseMode
from telegram.error import BadRequest

import chibi.services.scheduler_interface as scheduler_interface_module
from chibi.exceptions import ConfigurationError
from chibi.models import User
from chibi.schemas.app import ChatResponseSchema
from chibi.services.bot import handle_scheduler_trigger
from chibi.services.scheduler_interface import SchedulerInterface, _get_scheduler_bot
from chibi.services.user import get_llm_chat_completion_answer
from chibi.storage.local import LocalStorage

pytestmark = pytest.mark.asyncio


@pytest.fixture
def local_db(tmp_path: Path) -> LocalStorage:
    """Local storage backend pointed at a temp directory."""
    return LocalStorage(storage_path=str(tmp_path))


@pytest.fixture(autouse=True)
def _reset_scheduler_bot_singleton():
    """Reset the shared scheduler Bot singleton around every test."""
    scheduler_interface_module._scheduler_bot = None
    yield
    scheduler_interface_module._scheduler_bot = None


def _make_provider_mock(answer: str) -> MagicMock:
    """Build a mock LLM provider returning the given answer.

    Args:
        answer: Text the mocked provider should answer with.

    Returns:
        MagicMock with an awaited get_chat_response returning a chat response and no extra messages.
    """
    provider = MagicMock()
    provider.get_chat_response = AsyncMock(
        return_value=(ChatResponseSchema(answer=answer, provider="OpenAI", model="m", usage=None), [])
    )
    return provider


def _patch_active_llm(provider_mock: MagicMock):
    """Patch User's active LLM resolution to always yield the given provider mock.

    Args:
        provider_mock: Provider mock to return from get_active_llm_provider.

    Returns:
        Context manager patching both provider and model resolution on User.
    """
    return (
        patch.object(User, "get_active_llm_provider", new=MagicMock(return_value=provider_mock)),
        patch.object(User, "get_active_llm_model", new=MagicMock(return_value=None)),
    )


async def test_scheduler_trigger_reaches_llm_chain_with_guardrail(local_db: LocalStorage) -> None:
    """A scheduler trigger goes through the LLM chain as the third-type system message."""
    provider_mock = _make_provider_mock("All checks passed")
    provider_patch, model_patch = _patch_active_llm(provider_mock)
    interface = SchedulerInterface(user_id=1, storage_id=1, chat_id=1, thread_id=42)

    with provider_patch, model_patch:
        response = await get_llm_chat_completion_answer.__wrapped__(
            local_db,
            storage_id=1,
            interface=interface,
            scheduler_trigger_text="Check the reports",
            scheduler_job_id="agent:1:health",
        )

    assert response.answer == "All checks passed"
    kwargs = provider_mock.get_chat_response.await_args.kwargs
    payload = json.loads(kwargs["messages"][-1].content)
    assert payload["type"] == "scheduled_trigger"
    assert payload["trigger_text"] == "Check the reports"
    assert payload["job_id"] == "agent:1:health"
    assert "NOT a user message" in payload["desc"]
    assert "ACK" in payload["desc"]

    user = await local_db.get_or_create_user(user_id=1)
    history_contents = [message.content for message in user.thread_messages_map[42]]
    assert any("scheduled_trigger" in content for content in history_contents)


async def test_scheduler_trigger_without_text_raises_value_error(local_db: LocalStorage) -> None:
    """A trigger with empty text and no other prompt data is rejected."""
    interface = SchedulerInterface(user_id=1, storage_id=1, chat_id=1, thread_id=0)

    with pytest.raises(ValueError, match="No prompt data provided"):
        await get_llm_chat_completion_answer.__wrapped__(
            local_db, storage_id=1, interface=interface, scheduler_job_id="agent:1:job"
        )


async def test_ack_answer_is_not_delivered() -> None:
    """An agent answer that is a pure ACK marker stays silent."""
    interface = SchedulerInterface(user_id=1, storage_id=1, chat_id=1, thread_id=0)
    ack_response = ChatResponseSchema(answer="<chibi>ACK</chibi>", provider="OpenAI", model="m", usage=None)

    with (
        patch.object(SchedulerInterface, "send_message", new=AsyncMock()) as send_mock,
        patch(
            "chibi.services.bot.get_llm_chat_completion_answer",
            new=AsyncMock(return_value=ack_response),
        ),
        patch("chibi.services.bot.check_history_and_summarize", new=AsyncMock(return_value=False)),
    ):
        await handle_scheduler_trigger(trigger_text="tick", job_id="agent:1:job", interface=interface)

    send_mock.assert_not_awaited()


async def test_regular_answer_is_delivered_to_job_context() -> None:
    """A regular answer is delivered through the interface bound to the job context."""
    interface = SchedulerInterface(user_id=7, storage_id=7, chat_id=555, thread_id=99)
    answer_response = ChatResponseSchema(answer="Report ready", provider="OpenAI", model="m", usage=None)

    with (
        patch.object(SchedulerInterface, "send_message", new=AsyncMock()) as send_mock,
        patch(
            "chibi.services.bot.get_llm_chat_completion_answer",
            new=AsyncMock(return_value=answer_response),
        ),
        patch("chibi.services.bot.check_history_and_summarize", new=AsyncMock(return_value=False)),
    ):
        await handle_scheduler_trigger(trigger_text="report", job_id="agent:7:job", interface=interface)

    send_mock.assert_awaited_once_with(message="Report ready")
    assert interface.storage_id == 7
    assert interface.chat_id == 555
    assert interface.thread_id == 99


async def test_scheduler_interface_delivers_via_bot_to_chat_and_thread() -> None:
    """SchedulerInterface delivers messages through the shared Bot to the fixed chat/thread."""
    interface = SchedulerInterface(user_id=1, storage_id=1, chat_id=555, thread_id=99)
    bot_mock = MagicMock()
    bot_mock.send_message = AsyncMock()

    with patch("chibi.services.scheduler_interface._get_scheduler_bot", new=AsyncMock(return_value=bot_mock)):
        await interface.send_message("Hello")

    bot_mock.send_message.assert_awaited_once()
    kwargs = bot_mock.send_message.await_args.kwargs
    assert kwargs["chat_id"] == 555
    assert kwargs["message_thread_id"] == 99
    assert kwargs["parse_mode"] == ParseMode.MARKDOWN_V2


async def test_scheduler_interface_omits_thread_for_non_threaded_chat() -> None:
    """For thread_id 0 the message_thread_id argument is omitted (None)."""
    interface = SchedulerInterface(user_id=1, storage_id=1, chat_id=555, thread_id=0)
    bot_mock = MagicMock()
    bot_mock.send_message = AsyncMock()

    with patch("chibi.services.scheduler_interface._get_scheduler_bot", new=AsyncMock(return_value=bot_mock)):
        await interface.send_message("Hello")

    kwargs = bot_mock.send_message.await_args.kwargs
    assert kwargs["chat_id"] == 555
    assert kwargs["message_thread_id"] is None


async def test_scheduler_interface_falls_back_to_plain_text_on_bad_request() -> None:
    """MarkdownV2 delivery failure falls back to plain text delivery."""
    interface = SchedulerInterface(user_id=1, storage_id=1, chat_id=555, thread_id=0)
    bot_mock = MagicMock()
    bot_mock.send_message = AsyncMock(side_effect=[BadRequest("can't parse entities"), None])

    with patch("chibi.services.scheduler_interface._get_scheduler_bot", new=AsyncMock(return_value=bot_mock)):
        await interface.send_message("Hello")

    assert bot_mock.send_message.await_count == 2
    fallback_kwargs = bot_mock.send_message.await_args_list[1].kwargs
    assert "parse_mode" not in fallback_kwargs
    assert fallback_kwargs["text"] == "Hello"


async def test_scheduler_bot_is_initialized_once_and_reused() -> None:
    """The shared Bot is created and initialized exactly once."""
    settings_stub = SimpleNamespace(token="TEST_TOKEN", proxy=None)
    with (
        patch("chibi.services.scheduler_interface.telegram_settings", settings_stub),
        patch("chibi.services.scheduler_interface.Bot") as bot_cls,
    ):
        bot_instance = bot_cls.return_value
        bot_instance.initialize = AsyncMock()

        first = await _get_scheduler_bot()
        second = await _get_scheduler_bot()

    assert first is second is bot_instance
    bot_instance.initialize.assert_awaited_once()
    bot_cls.assert_called_once_with(token="TEST_TOKEN", request=None)


async def test_scheduler_bot_requires_token() -> None:
    """A missing bot token raises ConfigurationError instead of a silent failure."""
    settings_stub = SimpleNamespace(token=None, proxy=None)
    with (
        patch("chibi.services.scheduler_interface.telegram_settings", settings_stub),
        pytest.raises(ConfigurationError),
    ):
        await _get_scheduler_bot()


async def test_scheduler_trigger_persists_ack_answer_to_history(local_db: LocalStorage) -> None:
    """Even a silent ACK answer is persisted to the thread history."""
    provider_mock = _make_provider_mock("<chibi>ACK</chibi>")
    provider_patch, model_patch = _patch_active_llm(provider_mock)
    interface = SchedulerInterface(user_id=2, storage_id=2, chat_id=2, thread_id=0)

    with provider_patch, model_patch:
        await get_llm_chat_completion_answer.__wrapped__(
            local_db,
            storage_id=2,
            interface=interface,
            scheduler_trigger_text="heartbeat",
            scheduler_job_id="agent:2:beat",
        )

    user = await local_db.get_or_create_user(user_id=2)
    history_contents = [message.content for message in user.messages]
    assert any("<chibi>ACK</chibi>" in content for content in history_contents)
