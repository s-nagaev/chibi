"""Lightweight user interface for delivering agent output from scheduler job context."""

import asyncio
from io import BytesIO
from typing import Any

import telegramify_markdown
from loguru import logger
from telegram import Bot
from telegram.constants import MessageLimit, ParseMode
from telegram.error import BadRequest
from telegram.request import HTTPXRequest

from chibi.config import telegram_settings
from chibi.exceptions import ConfigurationError
from chibi.services.interface import UserInterface
from chibi.utils.telegram import split_markdown_v2

_scheduler_bot: Bot | None = None
_scheduler_bot_lock = asyncio.Lock()


async def _get_scheduler_bot() -> Bot:
    """Return the shared Bot instance used by SchedulerInterface, initializing it lazily.

    Returns:
        The shared Telegram Bot instance.

    Raises:
        ConfigurationError: If no Telegram bot token is configured.
    """
    global _scheduler_bot
    if _scheduler_bot is None:
        async with _scheduler_bot_lock:
            if _scheduler_bot is None:
                if not telegram_settings.token:
                    raise ConfigurationError("SchedulerInterface requires a configured Telegram bot token.")
                request = HTTPXRequest(proxy=telegram_settings.proxy) if telegram_settings.proxy else None
                bot = Bot(token=telegram_settings.token, request=request)
                await bot.initialize()
                _scheduler_bot = bot
                logger.info("SchedulerInterface: shared Telegram bot initialized.")
    return _scheduler_bot


class SchedulerInterface(UserInterface):
    """Minimal UserInterface bound to a scheduler job's chat context.

    Unlike TelegramInterface, it is built directly from identifiers stored in
    the job payload instead of an incoming Telegram Update, so it can be used
    from scheduler-triggered turns (action type ``self``) where no update
    exists. The agent's answer is delivered to Telegram through a shared Bot
    instance using the explicit ``chat_id`` and ``message_thread_id`` fixed at
    job creation time.
    """

    def __init__(self, user_id: int, storage_id: int, chat_id: int, thread_id: int = 0) -> None:
        """Initialize the interface from scheduler job context identifiers.

        Args:
            user_id: Telegram user id of the job owner.
            storage_id: Storage key of the conversation (user_id for private
                chats, chat_id for groups/forum topics).
            chat_id: Telegram chat to deliver the answer to.
            thread_id: Telegram message thread to deliver the answer to
                (0 for non-threaded chats).
        """
        self._user_id = user_id
        self._storage_id = storage_id
        self._chat_id = chat_id
        self._thread_id = thread_id

    @property
    def chat_id(self) -> str | int:
        """Returns the Telegram chat fixed in the job context.

        Returns:
            The chat identifier.
        """
        return self._chat_id

    @property
    def user_id(self) -> int:
        """Returns the Telegram user who owns the job.

        Returns:
            The user identifier.
        """
        return self._user_id

    @property
    def storage_id(self) -> int:
        """Returns the storage key of the conversation the job is bound to.

        Returns:
            The storage identifier.
        """
        return self._storage_id

    @property
    def thread_id(self) -> int:
        """Returns the Telegram message thread fixed in the job context.

        Returns:
            The thread identifier, or 0 for non-threaded chats.
        """
        return self._thread_id

    @property
    def user_data(self) -> str:
        """Returns a string representation of the job owner.

        Returns:
            The user data string.
        """
        return f"User #{self._user_id} (scheduler)"

    @property
    def chat_data(self) -> str:
        """Returns a string representation of the target chat context.

        Returns:
            The chat data string.
        """
        return f"chat #{self._chat_id}, thread #{self._thread_id}"

    @property
    def attached_document(self) -> dict[str, str] | None:
        """Returns the attached document data if present, otherwise None.

        Returns:
            None (there is no incoming message in scheduler context).
        """
        return None

    @property
    def attached_document_caption(self) -> str | None:
        """Returns the caption of the attached document if present.

        Returns:
            None (there is no incoming message in scheduler context).
        """
        return None

    async def get_text_prompt(self) -> str | None:
        """Retrieves the text prompt from the current message.

        Returns:
            None (there is no incoming user message in scheduler context).
        """
        return None

    async def get_voice_prompt(self) -> BytesIO | None:
        """Retrieves the voice prompt as a BytesIO object if present.

        Returns:
            None (voice input is not supported in scheduler context).
        """
        return None

    async def get_caption(self) -> str | None:
        """Retrieve the caption attached to the current message.

        Returns:
            None (attachments are not supported in scheduler context).
        """
        return None

    def set_caption(self, caption: str) -> None:
        """Store a caption for use during subsequent media sends.

        No-op: there is no incoming message to take a caption from.

        Args:
            caption: Ignored.
        """
        return None

    async def send_action_typing(self) -> None:
        """Sends a typing action to the user. No-op in scheduler context."""
        return None

    async def send_action_uploading_photo(self) -> None:
        """Sends an uploading photo action to the user. No-op in scheduler context."""
        return None

    async def send_action_recording(self) -> None:
        """Sends a recording voice action to the user. No-op in scheduler context."""
        return None

    async def send_reaction(self, reaction: str) -> None:
        """Sends a reaction to the user's message. No-op in scheduler context.

        Args:
            reaction: Ignored.
        """
        logger.debug(f"SchedulerInterface: reaction '{reaction}' skipped — no user message to react to.")
        return None

    async def delete_last_user_message(self) -> None:
        """Deletes the last message sent by the user. No-op in scheduler context."""
        return None

    async def send_message(self, message: str, reply: bool = True, **kwargs: Any) -> None:
        """Send a text message to the Telegram chat and thread fixed in the job context.

        The message is converted to MarkdownV2; if Telegram rejects it, it is
        re-sent in plain text chunks.

        Args:
            message: The text content to send.
            reply: Ignored — there is no incoming message to reply to.
            **kwargs: Additional arguments (ignored).
        """
        bot = await _get_scheduler_bot()
        message_thread_id = self._thread_id or None
        markdown_chunks = split_markdown_v2(telegramify_markdown.markdownify(message))
        try:
            for chunk in markdown_chunks:
                await bot.send_message(
                    chat_id=self._chat_id,
                    text=chunk,
                    parse_mode=ParseMode.MARKDOWN_V2,
                    message_thread_id=message_thread_id,
                )
        except BadRequest as e:
            logger.bind(user_id=self._user_id).warning(
                f"Failed to deliver scheduler message to {self.chat_data} as MarkdownV2 ({e}). Retrying in plain text."
            )
            plain_chunks = [
                message[i : i + MessageLimit.MAX_TEXT_LENGTH]
                for i in range(0, len(message), MessageLimit.MAX_TEXT_LENGTH)
            ]
            for chunk in plain_chunks:
                await bot.send_message(
                    chat_id=self._chat_id,
                    text=chunk,
                    message_thread_id=message_thread_id,
                )

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
        """Sends an audio file to the user.

        Args:
            audio: Ignored.
            reply: Ignored.
            title: Ignored.
            caption: Ignored.
            performer: Ignored.
            duration: Ignored.
            thumbnail: Ignored.
            filename: Ignored.
            **kwargs: Ignored.

        Raises:
            NotImplementedError: Media delivery is not supported in scheduler
                context in v1.
        """
        raise NotImplementedError("SchedulerInterface does not support audio delivery.")

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
        """Sends a video file to the user.

        Args:
            video: Ignored.
            reply: Ignored.
            title: Ignored.
            caption: Ignored.
            duration: Ignored.
            thumbnail: Ignored.
            filename: Ignored.
            **kwargs: Ignored.

        Raises:
            NotImplementedError: Media delivery is not supported in scheduler
                context in v1.
        """
        raise NotImplementedError("SchedulerInterface does not support video delivery.")

    async def send_images(self, images: list[BytesIO] | list[str], reply: bool = True, **kwargs: Any) -> None:
        """Sends a list of images to the user.

        Args:
            images: Ignored.
            reply: Ignored.
            **kwargs: Ignored.

        Raises:
            NotImplementedError: Media delivery is not supported in scheduler
                context in v1.
        """
        raise NotImplementedError("SchedulerInterface does not support image delivery.")

    async def send_document(
        self,
        document: bytes | BytesIO,
        filename: str | None = None,
        caption: str | None = None,
        thumbnail: bytes | None = None,
        **kwargs: Any,
    ) -> None:
        """Sends a document file to the user.

        Args:
            document: Ignored.
            filename: Ignored.
            caption: Ignored.
            thumbnail: Ignored.
            **kwargs: Ignored.

        Raises:
            NotImplementedError: Media delivery is not supported in scheduler
                context in v1.
        """
        raise NotImplementedError("SchedulerInterface does not support document delivery.")
