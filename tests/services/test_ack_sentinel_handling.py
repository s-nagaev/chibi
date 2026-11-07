"""Tests for ACK sentinel handling in the bot delivery paths and persistence."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chibi.models import Message, User
from chibi.schemas.app import ChatResponseSchema
from chibi.services.bot import handle_tool_response, handle_user_prompt
from chibi.services.user import get_llm_chat_completion_answer
from chibi.storage.local import LocalStorage
from chibi.utils.ack_sentinel import ACK_SENTINEL

pytestmark = pytest.mark.asyncio


def _make_chat_response(answer: str) -> ChatResponseSchema:
    """Build a chat response schema with the given answer.

    Args:
        answer: The agent's answer text.

    Returns:
        A ChatResponseSchema with no usage information.
    """
    return ChatResponseSchema(answer=answer, provider="OpenAI", model="m", usage=None)


def _make_interface() -> Any:
    """Build an interface stub covering both the tool and the user prompt paths.

    Returns:
        A SimpleNamespace stub with async mocks for every interface method the
        bot handlers touch.
    """
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


class TestToolPath:
    async def test_pure_ack_is_not_delivered(self) -> None:
        """A pure ACK answer on the tool-result path must not reach the user."""
        interface = _make_interface()
        with patch(
            "chibi.services.bot.get_llm_chat_completion_answer",
            new=AsyncMock(return_value=_make_chat_response(ACK_SENTINEL)),
        ):
            await handle_tool_response(tool_response=MagicMock(), interface=interface)

        interface.send_tool_answer.assert_not_awaited()

    async def test_mixed_answer_is_delivered_with_marker_stripped(self) -> None:
        """A mixed answer on the tool-result path is delivered without the marker."""
        interface = _make_interface()
        answer = f"Report is ready. {ACK_SENTINEL}"
        with patch(
            "chibi.services.bot.get_llm_chat_completion_answer", new=AsyncMock(return_value=_make_chat_response(answer))
        ):
            await handle_tool_response(tool_response=MagicMock(), interface=interface)

        interface.send_tool_answer.assert_awaited_once_with(content="Report is ready. ", model="m", provider="OpenAI")


class TestUserPath:
    async def test_pure_ack_is_not_delivered_and_reacted_to(self) -> None:
        """A pure ACK answer on the user path must not reach the user; a reaction is set."""
        interface = _make_interface()
        with patch(
            "chibi.services.bot.get_llm_chat_completion_answer",
            new=AsyncMock(return_value=_make_chat_response(ACK_SENTINEL)),
        ):
            await handle_user_prompt(interface=interface)

        interface.send_message.assert_not_awaited()
        interface.send_reaction.assert_awaited_once_with(reaction="👌")

    async def test_lowercase_marker_inside_length_budget_is_delivered(self) -> None:
        """The old fuzzy predicate swallowed this answer (contains marker, len <= 27); it must be delivered now."""
        interface = _make_interface()
        answer = "Done. <chibi>ack</chibi>"
        with patch(
            "chibi.services.bot.get_llm_chat_completion_answer", new=AsyncMock(return_value=_make_chat_response(answer))
        ):
            await handle_user_prompt(interface=interface)

        interface.send_message.assert_awaited_once_with(message=answer)
        interface.send_reaction.assert_not_awaited()

    async def test_mixed_answer_within_old_length_budget_is_delivered_stripped(self) -> None:
        """A mixed answer the old predicate would have swallowed is delivered with the marker stripped."""
        interface = _make_interface()
        answer = f"Done. {ACK_SENTINEL}"
        with patch(
            "chibi.services.bot.get_llm_chat_completion_answer", new=AsyncMock(return_value=_make_chat_response(answer))
        ):
            await handle_user_prompt(interface=interface)

        interface.send_message.assert_awaited_once_with(message="Done. ")
        interface.send_reaction.assert_not_awaited()

    async def test_marker_inside_longer_text_is_stripped_not_swallowed(self) -> None:
        """A long answer containing the marker is delivered with the marker removed."""
        interface = _make_interface()
        answer = f"Here is the analysis you asked for. {ACK_SENTINEL} Nothing else matters."
        with patch(
            "chibi.services.bot.get_llm_chat_completion_answer", new=AsyncMock(return_value=_make_chat_response(answer))
        ):
            await handle_user_prompt(interface=interface)

        interface.send_message.assert_awaited_once_with(
            message="Here is the analysis you asked for.  Nothing else matters."
        )


class TestPersistence:
    @pytest.fixture
    def local_db(self, tmp_path: Any) -> LocalStorage:
        """Local storage backend pointed at a temp directory.

        Args:
            tmp_path: Pytest temporary directory.

        Returns:
            A LocalStorage instance.
        """
        return LocalStorage(storage_path=str(tmp_path))

    async def test_mixed_answer_is_persisted_without_marker(self, local_db: LocalStorage) -> None:
        """The assistant message must be stripped before it reaches the storage layer."""
        mixed = f"Report ready. {ACK_SENTINEL}"
        provider_mock = MagicMock()
        provider_mock.get_chat_response = AsyncMock(
            return_value=(
                _make_chat_response(mixed),
                [Message(role="assistant", content=mixed)],
            )
        )
        interface = _make_interface()

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

        assert response.answer == "Report ready. "
        user = await local_db.get_or_create_user(user_id=1)
        assistant_contents = [message.content for message in user.messages if message.role == "assistant"]
        assert assistant_contents == ["Report ready. "]

    async def test_pure_ack_is_persisted_intact(self, local_db: LocalStorage) -> None:
        """A pure ACK answer keeps the marker in history (persisted, never delivered)."""
        provider_mock = MagicMock()
        provider_mock.get_chat_response = AsyncMock(
            return_value=(
                _make_chat_response(ACK_SENTINEL),
                [Message(role="assistant", content=ACK_SENTINEL)],
            )
        )
        interface = _make_interface()

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

        assert response.answer == ACK_SENTINEL
        user = await local_db.get_or_create_user(user_id=1)
        assistant_contents = [message.content for message in user.messages if message.role == "assistant"]
        assert assistant_contents == [ACK_SENTINEL]
