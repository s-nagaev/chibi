"""LLM thoughts parity in the Telegram SchedulerInterface (tg-thinking drafts)."""

from unittest.mock import AsyncMock

import pytest

import chibi.services.scheduler_interface as scheduler_interface_module
from chibi.services.interface import UserInterface
from chibi.services.scheduler_interface import SchedulerInterface, StdioSchedulerInterface

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _reset_scheduler_bot_singleton():
    """Reset the shared scheduler Bot singleton around every test."""
    scheduler_interface_module._scheduler_bot = None
    yield
    scheduler_interface_module._scheduler_bot = None


@pytest.fixture
def scheduler_bot() -> AsyncMock:
    """Install an AsyncMock as the shared scheduler Bot for the duration of a test."""
    bot = AsyncMock()
    scheduler_interface_module._scheduler_bot = bot
    return bot


@pytest.fixture
def interface() -> SchedulerInterface:
    """A scheduler interface bound to a non-threaded chat."""
    return SchedulerInterface(user_id=1, storage_id=1, chat_id=555, thread_id=0)


def _draft_calls(bot: AsyncMock) -> list:
    """Return all ``do_api_request`` calls recorded on the mocked bot."""
    return [c for c in bot.mock_calls if c[0] == "do_api_request"]


class TestSendLlmThoughts:
    """send_llm_thoughts delivers an ephemeral tg-thinking draft via the scheduler bot."""

    async def test_sends_thinking_draft(self, interface, scheduler_bot) -> None:
        """The draft is sent with chat_id and a non-zero draft_id; no thread key for thread 0."""
        await interface.send_llm_thoughts("reasoning about the task")

        calls = _draft_calls(scheduler_bot)
        assert len(calls) == 1
        args, kwargs = calls[0][1], calls[0][2]
        assert args[0] == "sendRichMessageDraft"
        payload = kwargs["api_kwargs"]
        assert payload["chat_id"] == 555
        assert payload["draft_id"] == interface._thinking_draft_id
        assert payload["draft_id"] != 0
        assert payload["rich_message"]["blocks"] == [{"type": "thinking", "text": "reasoning about the task"}]
        # thread_id=0 must not produce a message_thread_id key
        assert "message_thread_id" not in payload

    async def test_uses_thread_id_when_set(self, scheduler_bot) -> None:
        """Non-zero thread ids are forwarded as message_thread_id."""
        interface = SchedulerInterface(user_id=1, storage_id=1, chat_id=555, thread_id=42)

        await interface.send_llm_thoughts("reasoning")

        payload = _draft_calls(scheduler_bot)[0][2]["api_kwargs"]
        assert payload["message_thread_id"] == 42

    async def test_draft_id_is_stable_across_calls(self, interface, scheduler_bot) -> None:
        """Consecutive thought updates reuse the same draft_id."""
        await interface.send_llm_thoughts("first chunk")
        await interface.send_llm_thoughts("second chunk")

        calls = _draft_calls(scheduler_bot)
        assert len(calls) == 2
        first_id = calls[0][2]["api_kwargs"]["draft_id"]
        second_id = calls[1][2]["api_kwargs"]["draft_id"]
        assert first_id == second_id == interface._thinking_draft_id

    @pytest.mark.parametrize("thoughts", ["", "No content"])
    async def test_empty_thoughts_send_nothing(self, interface, scheduler_bot, thoughts) -> None:
        """Empty thoughts and the 'No content' sentinel never hit the bot."""
        await interface.send_llm_thoughts(thoughts)

        scheduler_bot.do_api_request.assert_not_awaited()

    async def test_send_failure_is_swallowed_and_draft_id_reset(self, interface, scheduler_bot) -> None:
        """A failing draft send must not raise and must reset the draft id."""
        scheduler_bot.do_api_request.side_effect = RuntimeError("telegram down")

        await interface.send_llm_thoughts("reasoning")  # must not raise

        assert interface._thinking_draft_id is None

    async def test_no_plain_text_fallback(self, interface, scheduler_bot) -> None:
        """Thoughts are never delivered as a plain text message on failure."""
        scheduler_bot.do_api_request.side_effect = RuntimeError("telegram down")

        await interface.send_llm_thoughts("reasoning")

        scheduler_bot.send_message.assert_not_awaited()


class TestClearThinkingDraft:
    """The draft is cleared before the final message and always reset."""

    async def test_send_message_clears_draft_before_final_message(self, interface, scheduler_bot) -> None:
        """The clear payload (zero-width space) precedes the final text message."""
        await interface.send_llm_thoughts("reasoning")
        draft_id = interface._thinking_draft_id

        await interface.send_message("final answer")

        calls = _draft_calls(scheduler_bot)
        assert len(calls) == 2
        clear_payload = calls[1][2]["api_kwargs"]
        assert clear_payload["draft_id"] == draft_id
        assert clear_payload["rich_message"]["blocks"] == [{"type": "thinking", "text": "\u200b"}]
        assert interface._thinking_draft_id is None
        # The clear draft call must precede the final send_message call.
        send_index = [c[0] for c in scheduler_bot.mock_calls].index("send_message")
        clear_index = [c[0] for c in scheduler_bot.mock_calls].index("do_api_request", 1)
        assert clear_index < send_index

    async def test_send_message_without_draft_makes_no_extra_api_call(self, interface, scheduler_bot) -> None:
        """send_message alone never calls do_api_request."""
        await interface.send_message("just a message")

        scheduler_bot.do_api_request.assert_not_awaited()
        scheduler_bot.send_message.assert_awaited_once()

    async def test_clear_failure_is_swallowed_and_draft_id_reset(self, interface, scheduler_bot) -> None:
        """A failing clear must not raise and must still reset the draft id."""
        await interface.send_llm_thoughts("reasoning")
        scheduler_bot.do_api_request.side_effect = RuntimeError("telegram down")

        await interface.send_message("final answer")  # must not raise

        assert interface._thinking_draft_id is None
        scheduler_bot.send_message.assert_awaited_once()


class TestStdioSchedulerInterfaceUnaffected:
    """The stdio scheduler interface gains no thinking-draft machinery."""

    def test_no_new_methods(self) -> None:
        """StdioSchedulerInterface inherits the base send_llm_thoughts unchanged."""
        assert not hasattr(StdioSchedulerInterface, "_thinking_draft_id")
        assert not hasattr(StdioSchedulerInterface, "_clear_thinking_draft")
        # send_llm_thoughts exists on the ABC base — it must not be overridden.
        assert StdioSchedulerInterface.send_llm_thoughts is UserInterface.send_llm_thoughts
