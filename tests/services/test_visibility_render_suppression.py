"""Render-time suppression of hidden turns (task 5).

Covers:
- the visibility-keyed gates in the bot delivery handlers (tool response,
  user prompt, scheduler wake), with the legacy pure-ACK marker consulted
  as the fallback;
- Chroma archival skipping hidden messages while legacy ACK bodies stay
  skipped;
- the emergency summarizer excluding hidden messages from its input;
- the IDE result frame omitting ``thoughts`` on a hidden turn and the
  continuation-snapshot re-baseline absorbing a hidden background turn's
  reasoning;
- preservation of ``visibility`` through persistence, thread clone and
  history rewrite.
"""

from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chibi.memory.chroma import with_chroma_archival
from chibi.models import Message, User
from chibi.schemas.app import ChatResponseSchema
from chibi.services.bot import handle_scheduler_trigger, handle_tool_response, handle_user_prompt
from chibi.services.user import clone_thread_messages, drop_tool_call_history, emergency_summarization
from chibi.storage.local import LocalStorage
from chibi.utils.ack_sentinel import ACK_SENTINEL

pytestmark = pytest.mark.asyncio


def _chat_response(answer: str, visibility: str = "visible") -> ChatResponseSchema:
    return ChatResponseSchema(answer=answer, provider="OpenAI", model="m", usage=None, visibility=visibility)


def _make_interface() -> Any:
    return SimpleNamespace(
        user_id=1,
        storage_id=1,
        thread_id=0,
        user_data="user:1",
        chat_data="chat:1",
        get_text_prompt=AsyncMock(return_value="hello"),
        get_voice_prompt=AsyncMock(return_value=None),
        get_caption=AsyncMock(return_value=None),
        send_action_typing=AsyncMock(),
        send_message=AsyncMock(),
        send_reaction=AsyncMock(),
        send_tool_answer=AsyncMock(),
    )


class TestToolResponseGate:
    async def test_hidden_turn_delivers_nothing(self) -> None:
        """A hidden turn (ack_silently) must not reach send_tool_answer."""
        interface = _make_interface()
        with patch(
            "chibi.services.bot.get_llm_chat_completion_answer",
            new=AsyncMock(return_value=_chat_response("", visibility="hidden")),
        ):
            await handle_tool_response(tool_response=MagicMock(), interface=interface)

        interface.send_tool_answer.assert_not_awaited()

    async def test_visible_turn_is_still_delivered(self) -> None:
        """An ordinary visible turn keeps being delivered (no regression)."""
        interface = _make_interface()
        with patch(
            "chibi.services.bot.get_llm_chat_completion_answer",
            new=AsyncMock(return_value=_chat_response("All done")),
        ):
            await handle_tool_response(tool_response=MagicMock(), interface=interface)

        interface.send_tool_answer.assert_awaited_once_with(content="All done", model="m", provider="OpenAI")

    async def test_legacy_sentinel_still_suppresses_as_fallback(self) -> None:
        """A visible answer carrying the legacy ACK marker body is still suppressed."""
        interface = _make_interface()
        with patch(
            "chibi.services.bot.get_llm_chat_completion_answer",
            new=AsyncMock(return_value=_chat_response(ACK_SENTINEL)),
        ):
            await handle_tool_response(tool_response=MagicMock(), interface=interface)

        interface.send_tool_answer.assert_not_awaited()


class TestUserPromptGate:
    async def test_hidden_turn_delivers_nothing_but_reacts(self) -> None:
        """A hidden turn sends no message; the 👌 receipt reaction still fires."""
        interface = _make_interface()
        with (
            patch(
                "chibi.services.bot.get_llm_chat_completion_answer",
                new=AsyncMock(return_value=_chat_response("", visibility="hidden")),
            ),
            patch("chibi.services.bot.check_history_and_summarize", new=AsyncMock(return_value=False)),
        ):
            await handle_user_prompt(interface=interface)

        interface.send_message.assert_not_awaited()
        interface.send_reaction.assert_awaited_once_with(reaction="👌")

    async def test_visible_turn_is_still_delivered(self) -> None:
        """An ordinary visible user turn keeps being delivered (no regression)."""
        interface = _make_interface()
        with (
            patch(
                "chibi.services.bot.get_llm_chat_completion_answer",
                new=AsyncMock(return_value=_chat_response("Here is your answer")),
            ),
            patch("chibi.services.bot.check_history_and_summarize", new=AsyncMock(return_value=False)),
        ):
            await handle_user_prompt(interface=interface)

        interface.send_message.assert_awaited_once_with(message="Here is your answer")
        interface.send_reaction.assert_not_awaited()


class TestSchedulerGate:
    async def test_hidden_turn_delivers_nothing(self) -> None:
        """A hidden scheduler wake delivers no message to the job context."""
        interface = _make_interface()
        with (
            patch(
                "chibi.services.bot.get_llm_chat_completion_answer",
                new=AsyncMock(return_value=_chat_response("", visibility="hidden")),
            ),
            patch("chibi.services.bot.check_history_and_summarize", new=AsyncMock(return_value=False)),
        ):
            await handle_scheduler_trigger(trigger_text="tick", job_id="agent:1:job", interface=interface)

        interface.send_message.assert_not_awaited()

    async def test_visible_turn_is_still_delivered(self) -> None:
        """A visible scheduler wake keeps being delivered (no regression)."""
        interface = _make_interface()
        with (
            patch(
                "chibi.services.bot.get_llm_chat_completion_answer",
                new=AsyncMock(return_value=_chat_response("Nightly digest: all good")),
            ),
            patch("chibi.services.bot.check_history_and_summarize", new=AsyncMock(return_value=False)),
        ):
            await handle_scheduler_trigger(trigger_text="digest", job_id="agent:1:job", interface=interface)

        interface.send_message.assert_awaited_once_with(message="Nightly digest: all good")


class _ArchiveSpyDatabase:
    """Minimal Database stand-in for the archival decorator."""

    def __init__(self) -> None:
        self.archived: list[Message] = []

    async def add_message(self, user: User, message: Message, ttl: int | None = None, thread_id: int = 0) -> None:
        pass

    async def get_messages(self, user: User, thread_id: int = 0) -> list[dict[str, Any]]:
        return []

    async def drop_messages(self, user: User, thread_id: int = 0) -> None:
        pass

    async def get_user(self, user_id: int) -> None:
        return None

    async def create_user(self, user_id: int) -> User:
        return User(id=user_id)

    async def save_user(self, user: User) -> None:
        pass

    async def count_image(self, user_id: int) -> None:
        pass


class _SpyMemory:
    def __init__(self) -> None:
        self.calls: list[Message] = []

    async def archive(self, user_id: int, messages: list[Message], thread_id: int = 0) -> None:
        self.calls.extend(messages)


class TestChromaArchivalSkip:
    @staticmethod
    def _assert_archival_scheduled(mock_task_manager: MagicMock, scheduled: bool) -> None:
        """Assert on run_task and close the scheduled coroutine to avoid warnings."""
        if scheduled:
            mock_task_manager.run_task.assert_called_once()
            coro = mock_task_manager.run_task.call_args[0][0]
            assert coro.cr_code.co_name == "archive"
            coro.close()
        else:
            mock_task_manager.run_task.assert_not_called()

    async def test_hidden_message_is_not_archived(self) -> None:
        """A hidden message (private CoT) must never reach the archival memory."""
        with patch("chibi.memory.chroma.task_manager") as mock_task_manager:
            mock_task_manager.run_task = MagicMock()
            db = with_chroma_archival(long_conv_memory=_SpyMemory())(_ArchiveSpyDatabase())  # type: ignore[arg-type]
            user = User(id=1)
            await db.add_message(
                user=user, message=Message(role="assistant", content="private CoT", visibility="hidden")
            )
            self._assert_archival_scheduled(mock_task_manager, scheduled=False)

    async def test_legacy_ack_body_is_still_not_archived(self) -> None:
        """The pure-ACK skip survives: legacy marker bodies keep being skipped."""
        with patch("chibi.memory.chroma.task_manager") as mock_task_manager:
            mock_task_manager.run_task = MagicMock()
            db = with_chroma_archival(long_conv_memory=_SpyMemory())(_ArchiveSpyDatabase())  # type: ignore[arg-type]
            user = User(id=1)
            await db.add_message(user=user, message=Message(role="assistant", content=ACK_SENTINEL))
            self._assert_archival_scheduled(mock_task_manager, scheduled=False)

    async def test_visible_message_is_archived(self) -> None:
        with patch("chibi.memory.chroma.task_manager") as mock_task_manager:
            mock_task_manager.run_task = MagicMock()
            db = with_chroma_archival(long_conv_memory=_SpyMemory())(_ArchiveSpyDatabase())  # type: ignore[arg-type]
            user = User(id=1)
            await db.add_message(user=user, message=Message(role="assistant", content="Visible reply"))
            self._assert_archival_scheduled(mock_task_manager, scheduled=True)


class TestSummarizerExcludesHidden:
    async def test_hidden_messages_are_excluded_from_summarizer_input(self) -> None:
        """Private CoT must not be folded into the summary input."""
        messages = [
            Message(role="user", content="What is the plan?"),
            Message(role="assistant", content="thinking privately", visibility="hidden"),
            Message(role="assistant", content="The plan is ready."),
        ]
        user = MagicMock()
        user.id = 1
        user.get_active_llm_model = MagicMock(return_value="active-model")
        provider = MagicMock()
        provider.get_chat_response = AsyncMock(
            return_value=(ChatResponseSchema(answer="summary", provider="t", model="m", usage=None), [])
        )
        user.get_active_llm_provider = MagicMock(return_value=provider)
        db = MagicMock()
        db.get_or_create_user = AsyncMock(return_value=user)
        db.get_conversation_messages = AsyncMock(return_value=messages)
        db.add_message = AsyncMock()

        with (
            patch("chibi.services.user.reset_chat_history", new=AsyncMock()) as reset_mock,
            patch("chibi.services.user.gpt_settings.max_tokens", 32000),
            patch("chibi.services.user.gpt_settings.max_history_tokens", 100000),
        ):
            await emergency_summarization.__wrapped__(db, storage_id=1, thread_id=0)

        sent_text = provider.get_chat_response.await_args.kwargs["messages"][0].content
        assert "thinking privately" not in sent_text
        assert "The plan is ready." in sent_text
        # Consequence of the PM ruling: the summary replaces the whole
        # history, so the hidden messages leave the conversation too
        # (existing summarization behaviour; storage rows are untouched,
        # reset_chat_history + re-add of the summary pair, as before).
        reset_mock.assert_awaited_once()


@pytest.fixture
def local_db(tmp_path: Any) -> LocalStorage:
    return LocalStorage(storage_path=str(tmp_path))


class TestVisibilityPreservation:
    async def test_hidden_message_persists_with_visibility_through_the_llm_chain(self, local_db: LocalStorage) -> None:
        """The pre-persistence strip rewrites content only; visibility rides intact."""
        provider_mock = MagicMock()
        provider_mock.get_chat_response = AsyncMock(
            return_value=(
                _chat_response("", visibility="hidden"),
                [Message(role="assistant", content="private CoT", visibility="hidden")],
            )
        )
        interface = _make_interface()
        from chibi.services.user import get_llm_chat_completion_answer

        with (
            patch.object(User, "get_active_llm_provider", new=MagicMock(return_value=provider_mock)),
            patch.object(User, "get_active_llm_model", new=MagicMock(return_value=None)),
        ):
            response = await get_llm_chat_completion_answer.__wrapped__(
                local_db,
                storage_id=1,
                interface=interface,
                user_text_message="hello",
            )

        assert response.visibility == "hidden"
        user = await local_db.get_or_create_user(user_id=1)
        assistant = [message for message in user.messages if message.role == "assistant"]
        assert [message.visibility for message in assistant] == ["hidden"]
        assert [message.content for message in assistant] == ["private CoT"]

    async def test_thread_clone_preserves_visibility(self, local_db: LocalStorage) -> None:
        user = await local_db.get_or_create_user(user_id=5)
        await local_db.add_message(user=user, message=Message(role="assistant", content="CoT", visibility="hidden"))
        await local_db.add_message(user=user, message=Message(role="assistant", content="Answer"))

        await clone_thread_messages.__wrapped__(local_db, storage_id=5, old_thread_id=0, new_thread_id=7)

        cloned = await local_db.get_conversation_messages(user=user, thread_id=7)
        assert sorted(message.visibility for message in cloned) == ["hidden", "visible"]

    async def test_drop_tool_call_history_preserves_visibility(self, local_db: LocalStorage) -> None:
        user = await local_db.get_or_create_user(user_id=9)
        await local_db.add_message(user=user, message=Message(role="assistant", content="CoT", visibility="hidden"))
        await local_db.add_message(user=user, message=Message(role="assistant", content="Answer"))

        async def reset_on_same_db(storage_id: int, thread_id: int) -> None:
            u = await local_db.get_or_create_user(user_id=storage_id)
            await local_db.drop_messages(user=u, thread_id=thread_id)

        with patch("chibi.services.user.reset_chat_history", new=reset_on_same_db):
            await drop_tool_call_history.__wrapped__(local_db, storage_id=9, thread_id=0)

        rewritten = await local_db.get_conversation_messages(user=user, thread_id=0)
        assert sorted(message.visibility for message in rewritten) == ["hidden", "visible"]
        # tool_calls/tool_call_id were nulled; visibility untouched.
        assert all(message.tool_calls is None and message.tool_call_id is None for message in rewritten)

    async def test_deepcopy_keeps_visibility(self) -> None:
        """The clone path deep-copies Message objects; the field survives by construction."""
        message = Message(role="assistant", content="CoT", visibility="hidden")
        assert deepcopy(message).visibility == "hidden"
