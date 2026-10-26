"""Tests for quoted-reply (Telegram "reply to") context propagation.

The user's flow under test:

1. the bot answers with several messages;
2. the user replies to one of them with "tell me more about that one";
3. the agent must see BOTH the follow-up prompt and the quoted message it
   answers — otherwise the follow-up is ambiguous.

The tests build real ``telegram`` objects (``Message``, ``Update``, ``User``,
``Chat``) instead of stand-ins: the interface reads a documented slice of the
Telegram API, and a fake would let the contract drift unnoticed.
"""

import datetime as dtm
import json
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from telegram import (
    Animation,
    Audio,
    Chat,
    Contact,
    Document,
    Location,
    Message,
    PhotoSize,
    Poll,
    PollOption,
    Sticker,
    TextQuote,
    Update,
    User,
    Venue,
    Video,
    VideoNote,
    Voice,
)
from telegram.ext import ContextTypes

import chibi.config  # noqa: F401
from chibi.schemas.app import ChatResponseSchema, UsageSchema
from chibi.services.interface import (
    _MEDIA_PLACEHOLDERS,
    _UNAVAILABLE_QUOTED_MESSAGE,
    TelegramInterface,
    UserInterface,
)
from chibi.services.user import get_llm_chat_completion_answer

_CHAT = Chat(id=12345, type=Chat.PRIVATE)
_NOW = dtm.datetime(2026, 1, 1, tzinfo=dtm.timezone.utc)


def _user(name: str = "test_name", is_bot: bool = False) -> User:
    """Build a Telegram user.

    Args:
        name: The user's display name.
        is_bot: Whether the user is a bot.

    Returns:
        The Telegram user object.
    """
    return User(id=1, first_name=name, is_bot=is_bot)


def _message(
    *,
    text: str | None = None,
    caption: str | None = None,
    from_user: User | None = None,
    reply_to_message: Message | None = None,
    quote: TextQuote | None = None,
    **media: Any,
) -> Message:
    """Build a real Telegram message.

    Args:
        text: Message text, if any.
        caption: Message caption, if any.
        from_user: The author of the message.
        reply_to_message: The quoted message, if any.
        quote: The partial quote, if any.
        **media: Media attributes to set (e.g. ``photo=PhotoSize(...)``).

    Returns:
        The Telegram message object.
    """
    # ``photo`` is a sequence in the Telegram API, the rest are scalars.
    if "photo" in media:
        media["photo"] = [media["photo"]]
    return Message(
        message_id=1,
        date=_NOW,
        chat=_CHAT,
        from_user=from_user,
        text=text,
        caption=caption,
        reply_to_message=reply_to_message,
        quote=quote,
        **media,
    )


def _interface(message: Message | None) -> TelegramInterface:
    """Build a TelegramInterface for an incoming message.

    Args:
        message: The incoming message, or None for an update without a message.

    Returns:
        The interface instance under test.
    """
    update = Update(update_id=1, message=message)
    return TelegramInterface(update=update, context=cast(ContextTypes.DEFAULT_TYPE, SimpleNamespace()))


class TestGetRepliedToPrompt:
    """TelegramInterface.get_replied_to_prompt content extraction."""

    @pytest.mark.asyncio
    async def test_no_reply_returns_none(self) -> None:
        """A plain message carries no quoted context."""
        assert await _interface(_message(text="text")).get_replied_to_prompt() is None

    @pytest.mark.asyncio
    async def test_missing_effective_message_returns_none(self) -> None:
        """An update without a usable message must not raise."""
        assert await _interface(None).get_replied_to_prompt() is None

    @pytest.mark.asyncio
    async def test_reply_to_bot_text_message(self) -> None:
        """The bot's quoted answer is described with its author and full text."""
        quoted = _message(text="1. test1\n2. test2\n3. test3", from_user=_user("Chibi", is_bot=True))

        assert await _interface(_message(text="text", reply_to_message=quoted)).get_replied_to_prompt() == (
            "Bot (Chibi): 1. test1 2. test2 3. test3"
        )

    @pytest.mark.asyncio
    async def test_reply_to_user_message_uses_user_name(self) -> None:
        """Quoted user messages are attributed to that user."""
        quoted = _message(text="test3", from_user=_user("test_name"))

        assert await _interface(_message(text="text", reply_to_message=quoted)).get_replied_to_prompt() == (
            "test_name: test3"
        )

    @pytest.mark.asyncio
    async def test_reply_to_media_caption(self) -> None:
        """A captioned media message is represented by its caption."""
        quoted = _message(
            caption="test",
            from_user=_user("Chibi", is_bot=True),
            photo=PhotoSize(file_id="f", file_unique_id="u", width=1, height=1),
        )

        assert await _interface(_message(text="text", reply_to_message=quoted)).get_replied_to_prompt() == (
            "Bot (Chibi): test"
        )

    @pytest.mark.parametrize(
        ("media", "placeholder"),
        [
            ({"photo": PhotoSize(file_id="f", file_unique_id="u", width=1, height=1)}, "[photo]"),
            ({"voice": Voice(file_id="f", file_unique_id="u", duration=1)}, "[voice message]"),
            ({"audio": Audio(file_id="f", file_unique_id="u", duration=1)}, "[audio]"),
            (
                {"video_note": VideoNote(file_id="f", file_unique_id="u", length=1, duration=1)},
                "[video message]",
            ),
            ({"video": Video(file_id="f", file_unique_id="u", width=1, height=1, duration=1)}, "[video]"),
            ({"document": Document(file_id="f", file_unique_id="u")}, "[document]"),
            (
                {
                    "sticker": Sticker(
                        file_id="f",
                        file_unique_id="u",
                        width=1,
                        height=1,
                        is_animated=False,
                        is_video=False,
                        type="regular",
                    )
                },
                "[sticker]",
            ),
            (
                {"animation": Animation(file_id="f", file_unique_id="u", width=1, height=1, duration=1)},
                "[GIF]",
            ),
            ({"location": Location(longitude=1.0, latitude=2.0)}, "[location]"),
            ({"contact": Contact(phone_number="+10000000000", first_name="n")}, "[contact]"),
            (
                {
                    "venue": Venue(
                        location=Location(longitude=1.0, latitude=2.0),
                        title="t",
                        address="a",
                    )
                },
                "[venue]",
            ),
            (
                {
                    "poll": Poll(
                        id="1",
                        question="q",
                        options=[PollOption(text="a", voter_count=0)],
                        total_voter_count=0,
                        is_closed=False,
                        is_anonymous=True,
                        type="regular",
                        allows_multiple_answers=False,
                    )
                },
                "[poll]",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_reply_to_textless_media_uses_placeholder(self, media: dict[str, Any], placeholder: str) -> None:
        """Every textless media kind has its own placeholder."""
        quoted = _message(from_user=_user("Chibi", is_bot=True), **media)

        assert await _interface(_message(text="text", reply_to_message=quoted)).get_replied_to_prompt() == (
            f"Bot (Chibi): {placeholder}"
        )

    @pytest.mark.asyncio
    async def test_every_media_placeholder_is_covered(self) -> None:
        """The parameterized test above covers every declared placeholder."""
        covered = {
            "[photo]",
            "[voice message]",
            "[audio]",
            "[video message]",
            "[video]",
            "[document]",
            "[sticker]",
            "[GIF]",
            "[location]",
            "[contact]",
            "[venue]",
            "[poll]",
        }
        declared = {placeholder for _, placeholder in _MEDIA_PLACEHOLDERS}

        assert declared == covered

    @pytest.mark.asyncio
    async def test_reply_without_author_is_attributed(self) -> None:
        """A quoted message without an author (e.g. a channel post) still renders."""
        quoted = _message(text="test")

        assert await _interface(_message(text="text", reply_to_message=quoted)).get_replied_to_prompt() == (
            "unknown author: test"
        )

    @pytest.mark.asyncio
    async def test_reply_without_any_content_uses_generic_placeholder(self) -> None:
        """A message with neither text nor media still renders."""
        quoted = _message(from_user=_user("Chibi", is_bot=True))

        assert await _interface(_message(text="text", reply_to_message=quoted)).get_replied_to_prompt() == (
            "Bot (Chibi): [non-text message]"
        )

    @pytest.mark.asyncio
    async def test_deleted_quoted_message_is_reported_as_unavailable(self) -> None:
        """Telegram reports deleted messages as an inaccessible message.

        PTB deserializes such a reply as a ``Message`` with a zero date and
        ``is_accessible`` set to False, so that is what the interface sees.
        """
        quoted = Message(message_id=7, date=dtm.datetime(1970, 1, 1, tzinfo=dtm.timezone.utc), chat=_CHAT)

        assert await _interface(_message(text="text", reply_to_message=quoted)).get_replied_to_prompt() == (
            _UNAVAILABLE_QUOTED_MESSAGE
        )

    @pytest.mark.asyncio
    async def test_deleted_quoted_message_from_raw_update_is_unavailable(self) -> None:
        """The same holds for a message parsed from a raw Telegram update."""
        update = Update.de_json(
            {
                "update_id": 1,
                "message": {
                    "message_id": 10,
                    "date": 1700000000,
                    "chat": {"id": 12345, "type": "private"},
                    "from": {"id": 2, "is_bot": False, "first_name": "U"},
                    "text": "follow up",
                    "reply_to_message": {
                        "message_id": 5,
                        "date": 0,
                        "chat": {"id": 12345, "type": "private"},
                    },
                },
            }
        )
        interface = TelegramInterface(update=update, context=cast(ContextTypes.DEFAULT_TYPE, SimpleNamespace()))

        assert await interface.get_replied_to_prompt() == _UNAVAILABLE_QUOTED_MESSAGE

    @pytest.mark.asyncio
    async def test_partial_quote_takes_precedence(self) -> None:
        """A quoted fragment is the most precise context and wins over the full message."""
        quoted = _message(text="1. test1\n2. test2", from_user=_user("Chibi", is_bot=True))
        quote = TextQuote(text="2. test2", position=7)

        assert await _interface(
            _message(text="more", reply_to_message=quoted, quote=quote)
        ).get_replied_to_prompt() == ("Bot (Chibi): 2. test2")

    @pytest.mark.asyncio
    async def test_partial_quote_without_quoted_message(self) -> None:
        """A fragment quote still carries the content when the parent is missing."""
        quote = TextQuote(text="quoted fragment", position=0)

        assert await _interface(_message(text="more", quote=quote)).get_replied_to_prompt() == (
            "unknown author: quoted fragment"
        )


class TestBaseInterfaceDefault:
    """The base UserInterface keeps transports without reply semantics unchanged."""

    @pytest.mark.asyncio
    async def test_base_interface_returns_none(self) -> None:
        """The default implementation reports no quoted context."""

        class MinimalUserInterface(UserInterface):
            pass

        assert await MinimalUserInterface().get_replied_to_prompt() is None


class _PlainInterface:
    """Minimal non-Telegram interface for prompt-shape assertions."""

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


async def _last_prompt(db: AsyncMock, fake_user: SimpleNamespace, **kwargs: Any) -> dict[str, Any]:
    """Run a turn through the real prompt builder and return the built prompt.

    Args:
        db: The mocked database.
        fake_user: The recording user fixture.
        **kwargs: Extra keyword arguments for ``get_llm_chat_completion_answer``.

    Returns:
        The parsed JSON prompt the provider received as the last message.
    """
    await get_llm_chat_completion_answer.__wrapped__(db, 1, cast(UserInterface, _PlainInterface()), **kwargs)
    message = _await_kwargs(fake_user.provider.get_chat_response)["messages"][-1]
    return json.loads(message.content)


def _await_kwargs(mock: AsyncMock) -> dict[str, Any]:
    """Return the keyword arguments of the most recent await on a mock.

    Args:
        mock: The async mock that was awaited.

    Returns:
        The keyword arguments the mock was awaited with.
    """
    call = mock.await_args
    assert call is not None
    return dict(call.kwargs)


def _mock_db(fake_user: SimpleNamespace) -> AsyncMock:
    """Build a database mock returning the given user and an empty history.

    Args:
        fake_user: The recording user fixture.

    Returns:
        The mocked database.
    """
    db = AsyncMock()
    db.get_or_create_user.return_value = fake_user
    db.get_conversation_messages.return_value = []
    return db


class TestRepliedToMessageInPrompt:
    """The quoted message reaches the LLM prompt only when it exists."""

    @pytest.mark.asyncio
    async def test_prompt_contains_quoted_message(self, fake_user: SimpleNamespace) -> None:
        """A replied turn carries both the new prompt and the quoted message."""
        db = _mock_db(fake_user)

        prompt = await _last_prompt(
            db,
            fake_user,
            user_text_message="test",
            replied_to_message="Bot (Chibi): 1. test1 2. test2 3. test3",
        )

        assert prompt["replied_to_message"] == "Bot (Chibi): 1. test1 2. test2 3. test3"
        assert prompt["prompt"] == "test"
        assert prompt["type"] == "user message"

    @pytest.mark.asyncio
    async def test_prompt_shape_unchanged_without_reply(self, fake_user: SimpleNamespace) -> None:
        """Without a quoted message the prompt keys are exactly the legacy ones."""
        db = _mock_db(fake_user)

        prompt = await _last_prompt(db, fake_user, user_text_message="hello")

        assert set(prompt) == {"user", "prompt", "datetime_now", "type", "transcribed_from_voice_message"}

    @pytest.mark.asyncio
    async def test_empty_reply_string_is_not_injected(self, fake_user: SimpleNamespace) -> None:
        """An empty quoted description must not add a meaningless key."""
        db = _mock_db(fake_user)

        prompt = await _last_prompt(db, fake_user, user_text_message="hello", replied_to_message="")

        assert "replied_to_message" not in prompt

    @pytest.mark.asyncio
    async def test_caption_turn_keeps_quoted_message(self, fake_user: SimpleNamespace) -> None:
        """A file upload sent as a reply keeps the quoted context.

        The user replies to a message and attaches a photo at the same time:
        the caption turn must still carry what the user was answering to.
        """
        db = _mock_db(fake_user)

        prompt = await _last_prompt(
            db,
            fake_user,
            user_caption='{"user_caption": "look"}',
            replied_to_message="Bot (Chibi): 1. test1",
        )

        assert prompt["type"] == "system message"
        assert prompt["replied_to_message"] == "Bot (Chibi): 1. test1"

    @pytest.mark.asyncio
    async def test_tool_response_turn_has_no_quoted_message(self, fake_user: SimpleNamespace) -> None:
        """A tool result is not a reply, so no quoted context is attached."""
        db = _mock_db(fake_user)

        prompt = await _last_prompt(
            db,
            fake_user,
            tool_message=cast(Any, SimpleNamespace(tool_name="tool", model_dump=lambda: {})),
            replied_to_message="Bot (Chibi): 1. test1",
        )

        assert prompt["type"] == "tool response"
        assert "replied_to_message" not in prompt

    @pytest.mark.asyncio
    async def test_scheduler_trigger_turn_has_no_quoted_message(self, fake_user: SimpleNamespace) -> None:
        """A scheduler trigger is not a reply, so no quoted context is attached."""
        db = _mock_db(fake_user)

        prompt = await _last_prompt(
            db,
            fake_user,
            scheduler_trigger_text="check the reports",
            scheduler_job_id="agent:1:job",
            replied_to_message="Bot (Chibi): 1. test1",
        )

        assert prompt["type"] == "scheduled_trigger"
        assert "replied_to_message" not in prompt


class _PromptInterface:
    """Interface stub exposing a fixed quoted message for the bot wiring."""

    user_data = "user-1"
    chat_data = "chat-1"
    storage_id = 1
    thread_id = 0

    def __init__(self, replied_to: str | None) -> None:
        """Store the quoted message the interface should report.

        Args:
            replied_to: The quoted message description, or None.
        """
        self._replied_to = replied_to

    async def get_text_prompt(self) -> str | None:
        """Return the typed prompt."""
        return "test"

    async def get_voice_prompt(self) -> None:
        """Report no voice prompt."""
        return None

    async def get_caption(self) -> str | None:
        """Report no caption."""
        return None

    async def get_replied_to_prompt(self) -> str | None:
        """Return the stored quoted message description."""
        return self._replied_to

    async def send_action_typing(self) -> None:
        """Do nothing."""
        return None

    async def send_message(self, message: str, **kwargs: Any) -> None:
        """Do nothing.

        Args:
            message: The message text.
            **kwargs: Extra arguments.
        """
        return None


class TestHandleUserPromptWiring:
    """handle_user_prompt forwards the quoted context to the prompt builder."""

    @pytest.mark.asyncio
    async def test_replied_prompt_is_forwarded(self) -> None:
        """The interface's quoted-message description reaches the LLM call."""
        from chibi.services.bot import handle_user_prompt

        chat_response = ChatResponseSchema(answer="Answer", provider="openai", model="gpt-4o", usage=None)
        llm_call = AsyncMock(return_value=chat_response)

        with (
            patch("chibi.services.bot.get_llm_chat_completion_answer", new=llm_call),
            patch("chibi.services.bot.check_history_and_summarize", new=AsyncMock(return_value=False)),
        ):
            await handle_user_prompt(interface=cast(UserInterface, _PromptInterface("Bot (Chibi): 1. test1 2. test2")))

        assert _await_kwargs(llm_call)["replied_to_message"] == "Bot (Chibi): 1. test1 2. test2"

    @pytest.mark.asyncio
    async def test_no_reply_forwards_none(self) -> None:
        """A plain message forwards None, leaving the prompt untouched."""
        from chibi.services.bot import handle_user_prompt

        chat_response = ChatResponseSchema(answer="Answer", provider="openai", model="gpt-4o", usage=None)
        llm_call = AsyncMock(return_value=chat_response)

        with (
            patch("chibi.services.bot.get_llm_chat_completion_answer", new=llm_call),
            patch("chibi.services.bot.check_history_and_summarize", new=AsyncMock(return_value=False)),
        ):
            await handle_user_prompt(interface=cast(UserInterface, _PromptInterface(None)))

        assert _await_kwargs(llm_call)["replied_to_message"] is None


class TestQuotedMessageLogging:
    """The quoted message is logged when prompt logging is enabled."""

    @pytest.mark.asyncio
    async def test_quoted_message_is_logged(self) -> None:
        """The quoted message appears in the logs when log_prompt_data is on."""
        from loguru import logger

        from chibi.services.bot import handle_user_prompt

        records: list[str] = []
        handler_id = logger.add(lambda message: records.append(message.record["message"]), level="INFO")
        chat_response = ChatResponseSchema(answer="Answer", provider="openai", model="gpt-4o", usage=None)

        try:
            with (
                patch("chibi.services.bot.get_llm_chat_completion_answer", new=AsyncMock(return_value=chat_response)),
                patch("chibi.services.bot.check_history_and_summarize", new=AsyncMock(return_value=False)),
                patch("chibi.services.bot.application_settings.log_prompt_data", new=True),
            ):
                await handle_user_prompt(
                    interface=cast(UserInterface, _PromptInterface("Bot (Chibi): 1. test1\n2. test2")),
                )
        finally:
            logger.remove(handler_id)

        assert any("replied to a message: Bot (Chibi): 1. test1 2. test2" in record for record in records)

    @pytest.mark.asyncio
    async def test_quoted_message_is_not_logged_when_disabled(self) -> None:
        """The quoted message stays out of the logs when log_prompt_data is off."""
        from loguru import logger

        from chibi.services.bot import handle_user_prompt

        records: list[str] = []
        handler_id = logger.add(lambda message: records.append(message.record["message"]), level="INFO")
        chat_response = ChatResponseSchema(answer="Answer", provider="openai", model="gpt-4o", usage=None)

        try:
            with (
                patch("chibi.services.bot.get_llm_chat_completion_answer", new=AsyncMock(return_value=chat_response)),
                patch("chibi.services.bot.check_history_and_summarize", new=AsyncMock(return_value=False)),
                patch("chibi.services.bot.application_settings.log_prompt_data", new=False),
            ):
                await handle_user_prompt(
                    interface=cast(UserInterface, _PromptInterface("Bot (Chibi): 1. test1")),
                )
        finally:
            logger.remove(handler_id)

        assert not any("replied to a message" in record for record in records)
