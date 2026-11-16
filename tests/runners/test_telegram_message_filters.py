"""Regression tests for the prompt and file_upload message filter chains."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

from telegram import Chat, Document, Message, MessageEntity, PhotoSize, Update, User, Voice
from telegram.ext import MessageHandler

from chibi.runners.telegram import ChibiBot

CHAT_ID = 42
BOT_USER_ID = 1
HUMAN_USER_ID = 7


def make_message(
    text: str | None = None,
    reply_to_bot: bool = False,
    voice: Voice | None = None,
    document: Document | None = None,
    photo: list[PhotoSize] | None = None,
) -> Message:
    """Build a real Message addressed to the bot.

    Args:
        text: Text body of the message, if any.
        reply_to_bot: Whether the message replies to a bot message.
        voice: Voice note attached to the message, if any.
        document: Document attached to the message, if any.
        photo: Photo sizes attached to the message, if any.

    Returns:
        The constructed Message object.
    """
    chat = Chat(id=CHAT_ID, type=Chat.PRIVATE)
    date = datetime(2026, 11, 15, 12, 0, tzinfo=UTC)
    reply_to_message = None
    if reply_to_bot:
        reply_to_message = Message(
            message_id=1,
            date=date,
            chat=chat,
            from_user=User(id=BOT_USER_ID, first_name="Chibi", is_bot=True),
        )
    return Message(
        message_id=2,
        date=date,
        chat=chat,
        from_user=User(id=HUMAN_USER_ID, first_name="Tester", is_bot=False),
        text=text,
        reply_to_message=reply_to_message,
        voice=voice,
        document=document,
        photo=photo,
    )


def make_command_message(text: str = "/start") -> Message:
    """Build a fresh bot-command message.

    Args:
        text: The command text, defaults to /start.

    Returns:
        The constructed Message object with a bot_command entity.
    """
    message = make_message(text=text)
    return Message(
        message_id=message.message_id,
        date=message.date,
        chat=message.chat,
        from_user=message.from_user,
        text=text,
        entities=[MessageEntity(type=MessageEntity.BOT_COMMAND, offset=0, length=len(text))],
    )


def make_update(message: Message, edited: bool) -> Update:
    """Wrap a message into a fresh or edited update.

    Args:
        message: The message carrying the update payload.
        edited: True wraps the message as edited_message, False as message.

    Returns:
        The constructed Update object.
    """
    if edited:
        return Update(update_id=1, edited_message=message)
    return Update(update_id=1, message=message)


def make_voice() -> Voice:
    """Build a minimal Voice object.

    Returns:
        The constructed Voice object.
    """
    return Voice(file_id="voice", file_unique_id="voice-unique", duration=2)


def make_document() -> Document:
    """Build a minimal Document object.

    Returns:
        The constructed Document object.
    """
    return Document(file_id="doc", file_unique_id="doc-unique", file_name="report.pdf")


def make_photo() -> list[PhotoSize]:
    """Build a minimal photo size list.

    Returns:
        The list with a single constructed PhotoSize object.
    """
    return [PhotoSize(file_id="photo", file_unique_id="photo-unique", width=10, height=10)]


class TestPromptMessageFilters:
    """Tests for the filter chain guarding the prompt handler."""

    def test_fresh_text_message_matches(self) -> None:
        """A fresh text message still triggers the prompt handler."""
        assert ChibiBot.prompt_message_filters.check_update(make_update(make_message(text="hello"), edited=False))

    def test_edited_text_message_does_not_match(self) -> None:
        """An edited text message no longer triggers the prompt handler."""
        assert not ChibiBot.prompt_message_filters.check_update(make_update(make_message(text="hello"), edited=True))

    def test_fresh_reply_to_bot_matches(self) -> None:
        """A fresh message replying to the bot still triggers the prompt handler."""
        update = make_update(make_message(text="hello", reply_to_bot=True), edited=False)
        assert ChibiBot.prompt_message_filters.check_update(update)

    def test_edited_reply_to_bot_does_not_match(self) -> None:
        """An edited message replying to the bot no longer triggers the prompt handler."""
        update = make_update(make_message(text="hello", reply_to_bot=True), edited=True)
        assert not ChibiBot.prompt_message_filters.check_update(update)

    def test_fresh_command_does_not_match(self) -> None:
        """A fresh command message stays excluded from the prompt handler."""
        assert not ChibiBot.prompt_message_filters.check_update(make_update(make_command_message(), edited=False))

    def test_fresh_voice_message_matches(self) -> None:
        """A fresh voice message still triggers the prompt handler."""
        update = make_update(make_message(voice=make_voice()), edited=False)
        assert ChibiBot.prompt_message_filters.check_update(update)

    def test_edited_voice_message_does_not_match(self) -> None:
        """An edited voice message no longer triggers the prompt handler."""
        update = make_update(make_message(voice=make_voice()), edited=True)
        assert not ChibiBot.prompt_message_filters.check_update(update)


class TestFileUploadMessageFilters:
    """Tests for the filter chain guarding the file_upload handler."""

    def test_fresh_document_message_matches(self) -> None:
        """A fresh document message still triggers the file_upload handler."""
        update = make_update(make_message(document=make_document()), edited=False)
        assert ChibiBot.file_upload_message_filters.check_update(update)

    def test_edited_document_message_does_not_match(self) -> None:
        """An edited document message no longer triggers the file_upload handler."""
        update = make_update(make_message(document=make_document()), edited=True)
        assert not ChibiBot.file_upload_message_filters.check_update(update)

    def test_fresh_photo_message_matches(self) -> None:
        """A fresh photo message still triggers the file_upload handler."""
        update = make_update(make_message(photo=make_photo()), edited=False)
        assert ChibiBot.file_upload_message_filters.check_update(update)

    def test_edited_photo_message_does_not_match(self) -> None:
        """An edited photo message no longer triggers the file_upload handler."""
        update = make_update(make_message(photo=make_photo()), edited=True)
        assert not ChibiBot.file_upload_message_filters.check_update(update)

    def test_fresh_command_with_document_does_not_match(self) -> None:
        """A fresh command message stays excluded from the file_upload handler."""
        chat = Chat(id=CHAT_ID, type=Chat.PRIVATE)
        date = datetime(2026, 11, 15, 12, 0, tzinfo=UTC)
        message = Message(
            message_id=2,
            date=date,
            chat=chat,
            from_user=User(id=HUMAN_USER_ID, first_name="Tester", is_bot=False),
            text="/start",
            entities=[MessageEntity(type=MessageEntity.BOT_COMMAND, offset=0, length=6)],
            document=make_document(),
        )
        assert not ChibiBot.file_upload_message_filters.check_update(make_update(message, edited=False))


class TestPromptHandlerWiring:
    """Tests that the registered handlers use the update-type-restricted filters."""

    def test_prompt_handler_rejects_edited_update(self) -> None:
        """The prompt MessageHandler as wired in run() ignores edited updates."""
        handler = MessageHandler(ChibiBot.prompt_message_filters, MagicMock())
        assert not handler.check_update(make_update(make_message(text="hello"), edited=True))
        assert handler.check_update(make_update(make_message(text="hello"), edited=False))
