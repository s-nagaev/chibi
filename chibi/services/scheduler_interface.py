"""Lightweight user interface for delivering agent output from scheduler job context."""

import asyncio
from io import BytesIO
from typing import Any, Awaitable, Callable

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

StdioDeliveryEmitter = Callable[[int, str], Awaitable[None]]

_stdio_delivery_emitter: StdioDeliveryEmitter | None = None


def set_stdio_delivery_emitter(emitter: StdioDeliveryEmitter) -> None:
    """Register the connected stdio session's scheduler delivery emitter.

    Called by the stdio runner at handshake so fired scheduler jobs can reach
    the connected client outside any request context. There is at most one
    stdio session per process, so a single module-level slot is sufficient;
    a fresh handshake overwrites any previous registration.

    Args:
        emitter: Async callable taking ``(thread_id, content)`` that delivers
            one ``message`` frame to the connected stdio client (capability
            check, stale-thread policy and stdout-lock serialization included).
    """
    global _stdio_delivery_emitter
    _stdio_delivery_emitter = emitter


def clear_stdio_delivery_emitter() -> None:
    """Unregister the scheduler delivery emitter (stdio session is gone)."""
    global _stdio_delivery_emitter
    _stdio_delivery_emitter = None


def get_stdio_delivery_emitter() -> StdioDeliveryEmitter | None:
    """Return the registered stdio session delivery emitter, if any.

    Returns:
        The emitter registered by the stdio runner at handshake, or None when
        no stdio session is connected (e.g. the Telegram process, where the
        scheduler never delivers through stdio).
    """
    return _stdio_delivery_emitter


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

    uses_uploaded_file_storage = False

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
            caption: The caption text to associate with the next media message.

        Note:
            No-op in scheduler context: there is no incoming message to take
            a caption from.
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
            reaction: The reaction to send.
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
            reply: Whether to reply to the user's message; has no effect in
                scheduler context because there is no incoming message to
                reply to.
            **kwargs: Additional arguments for the message sending function.
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
            audio: The audio data or path to send.
            reply: Whether to reply to the user's message.
            title: The title of the audio.
            caption: The caption for the audio.
            performer: The performer of the audio.
            duration: The duration of the audio in seconds.
            thumbnail: The thumbnail data for the audio.
            filename: The filename for the audio.
            **kwargs: Additional arguments for the audio sending function.

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
            video: The video data or path to send.
            reply: Whether to reply to the user's message.
            title: The title of the video.
            caption: The caption for the video.
            duration: The duration of the video in seconds.
            thumbnail: The thumbnail data for the video.
            filename: The filename for the video.
            **kwargs: Additional arguments for the video sending function.

        Raises:
            NotImplementedError: Media delivery is not supported in scheduler
                context in v1.
        """
        raise NotImplementedError("SchedulerInterface does not support video delivery.")

    async def send_images(self, images: list[BytesIO] | list[str], reply: bool = True, **kwargs: Any) -> None:
        """Sends a list of images to the user.

        Args:
            images: A list of image data or paths to send.
            reply: Whether to reply to the user's message.
            **kwargs: Additional arguments for the image sending function.

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
            document: The document data to send.
            filename: The filename for the document.
            caption: The caption for the document.
            thumbnail: The thumbnail data for the document.
            **kwargs: Additional arguments for the document sending function.

        Raises:
            NotImplementedError: Media delivery is not supported in scheduler
                context in v1.
        """
        raise NotImplementedError("SchedulerInterface does not support document delivery.")


class StdioSchedulerInterface(UserInterface):
    """Minimal UserInterface delivering scheduler output to a stdio client.

    The stdio counterpart of :class:`SchedulerInterface`: built directly from
    identifiers stored in the job payload, but instead of a Telegram bot the
    answer is handed to the session-level delivery emitter that the stdio
    runner registered at handshake, so self-wake answers, ``notify`` messages
    and failure notes reach the connected client as unsolicited ``message``
    frames. Delivery is best effort: when no stdio session is connected, or
    the client did not declare the ``background_messages`` capability, or the
    payload thread no longer exists in the current session, the message is
    logged and dropped by the emitter — the job itself still ran.
    """

    uses_uploaded_file_storage = False

    def __init__(self, user_id: int, storage_id: int, chat_id: int, thread_id: int = 0) -> None:
        """Initialize the interface from scheduler job context identifiers.

        Args:
            user_id: Integer id of the job owner (the negative reserved
                ``IDE_STORAGE_ID`` for IDE/stdio sessions).
            storage_id: Storage key of the conversation (the negative reserved
                ``IDE_STORAGE_ID`` for IDE/stdio sessions).
            chat_id: Chat the job delivers its output to (unused on stdio,
                kept for payload parity with the Telegram interface).
            thread_id: Client-minted session thread the job is bound to.
        """
        self._user_id = user_id
        self._storage_id = storage_id
        self._chat_id = chat_id
        self._thread_id = thread_id

    @property
    def chat_id(self) -> str | int:
        """Returns the chat fixed in the job context.

        Returns:
            The chat identifier.
        """
        return self._chat_id

    @property
    def user_id(self) -> int:
        """Returns the user who owns the job.

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
        """Returns the session thread fixed in the job context.

        Returns:
            The thread identifier.
        """
        return self._thread_id

    @property
    def user_data(self) -> str:
        """Returns a string representation of the job owner.

        Returns:
            The user data string.
        """
        return f"User #{self._user_id} (stdio scheduler)"

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
            caption: The caption text to associate with the next media message.

        Note:
            No-op in scheduler context: there is no incoming message to take
            a caption from.
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
            reaction: The reaction to send.
        """
        logger.debug(f"StdioSchedulerInterface: reaction '{reaction}' skipped — no user message to react to.")
        return None

    async def delete_last_user_message(self) -> None:
        """Deletes the last message sent by the user. No-op in scheduler context."""
        return None

    async def send_message(self, message: str, reply: bool = True, **kwargs: Any) -> None:
        """Deliver a text message to the connected stdio client.

        The message goes through the session-level emitter registered by the
        stdio runner at handshake and reaches the client as an unsolicited
        ``message`` frame (capability-gated, stdout-lock serialized, skipped
        for threads unknown to the current session). When no stdio session is
        connected, the message is logged and dropped.

        Args:
            message: The text content to send.
            reply: Whether to reply to the user's message; has no effect in
                scheduler context because there is no incoming message to
                reply to.
            **kwargs: Additional arguments for the message sending function.
        """
        emitter = get_stdio_delivery_emitter()
        if emitter is None:
            logger.bind(user_id=self._user_id).debug(
                f"StdioSchedulerInterface: message for {self.chat_data} dropped — "
                f"no stdio session emitter is registered."
            )
            return
        await emitter(self._thread_id, message)

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
            audio: The audio data or path to send.
            reply: Whether to reply to the user's message.
            title: The title of the audio.
            caption: The caption for the audio.
            performer: The performer of the audio.
            duration: The duration of the audio.
            thumbnail: The thumbnail data for the audio.
            filename: The filename for the audio.
            **kwargs: Additional arguments for the audio sending function.

        Raises:
            NotImplementedError: Media delivery is not supported in scheduler
                context in v1.
        """
        raise NotImplementedError("StdioSchedulerInterface does not support audio delivery.")

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
            video: The video data or path to send.
            reply: Whether to reply to the user's message.
            title: The title of the video.
            caption: The caption for the video.
            duration: The duration of the video.
            thumbnail: The thumbnail data for the video.
            filename: The filename for the video.
            **kwargs: Additional arguments for the video sending function.

        Raises:
            NotImplementedError: Media delivery is not supported in scheduler
                context in v1.
        """
        raise NotImplementedError("StdioSchedulerInterface does not support video delivery.")

    async def send_images(self, images: list[BytesIO] | list[str], reply: bool = True, **kwargs: Any) -> None:
        """Sends a list of images to the user.

        Args:
            images: A list of image data or paths to send.
            reply: Whether to reply to the user's message.
            **kwargs: Additional arguments for the image sending function.

        Raises:
            NotImplementedError: Media delivery is not supported in scheduler
                context in v1.
        """
        raise NotImplementedError("StdioSchedulerInterface does not support image delivery.")

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
            document: The document data to send.
            filename: The filename for the document.
            caption: The caption for the document.
            thumbnail: The thumbnail data for the document.
            **kwargs: Additional arguments for the document sending function.

        Raises:
            NotImplementedError: Media delivery is not supported in scheduler
                context in v1.
        """
        raise NotImplementedError("StdioSchedulerInterface does not support document delivery.")
