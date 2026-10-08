"""Lightweight user interface for delivering agent output from scheduler job context."""

import asyncio
import random
from io import BytesIO
from typing import Any, Awaitable, Callable

import telegramify_markdown
from loguru import logger
from telegram import Bot, InputMediaDocument, InputMediaPhoto
from telegram.constants import ChatAction, FileSizeLimit, MessageLimit, ParseMode
from telegram.error import BadRequest
from telegram.request import HTTPXRequest

from chibi.config import telegram_settings
from chibi.constants import AUDIO_UPLOAD_TIMEOUT, FILE_UPLOAD_TIMEOUT, IMAGE_UPLOAD_TIMEOUT
from chibi.exceptions import ConfigurationError
from chibi.services.interface import UserInterface
from chibi.utils.rich_message import RichMessageBuilder
from chibi.utils.telegram import download_image, split_markdown_v2

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
        self._thinking_draft_id: int | None = None

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

    async def _send_chat_action(self, action: ChatAction) -> None:
        """Send a chat action through the shared scheduler Bot, swallowing failures.

        Chat actions are purely cosmetic status indicators; a delivery failure
        must never abort the scheduled job, so any exception is logged as a
        warning and discarded (deliberately different from TelegramInterface,
        which lets the exception propagate).

        Args:
            action: The Telegram chat action to send.
        """
        bot = await _get_scheduler_bot()
        try:
            await bot.send_chat_action(
                chat_id=self._chat_id,
                action=action,
                message_thread_id=self._thread_id or None,
            )
        except Exception as e:
            logger.bind(user_id=self._user_id).warning(
                f"Failed to send chat action '{action}' to {self.chat_data}: {e}. "
                f"Chat actions are cosmetic and never fail the scheduled job."
            )

    async def send_action_typing(self) -> None:
        """Sends a typing action to the chat fixed in the job context.

        Best effort: failures are logged and swallowed because a chat action
        is cosmetic and must never fail the scheduled job.
        """
        await self._send_chat_action(ChatAction.TYPING)

    async def send_action_uploading_photo(self) -> None:
        """Sends an uploading photo action to the chat fixed in the job context.

        Best effort: failures are logged and swallowed because a chat action
        is cosmetic and must never fail the scheduled job.
        """
        await self._send_chat_action(ChatAction.UPLOAD_PHOTO)

    async def send_action_recording(self) -> None:
        """Sends a recording voice action to the chat fixed in the job context.

        Best effort: failures are logged and swallowed because a chat action
        is cosmetic and must never fail the scheduled job.
        """
        await self._send_chat_action(ChatAction.RECORD_VOICE)

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

    async def _clear_thinking_draft(self) -> None:
        """Clear any active ``<tg-thinking>`` draft before the final message.

        Mirrors ``TelegramInterface._clear_thinking_draft``: works around a
        known Mac Telegram client bug where the ``<tg-thinking>`` draft
        persists and overlaps the final message. Delivery goes through the
        shared scheduler Bot; a failure is logged and discarded because the
        draft is cosmetic and must never fail the scheduled job.
        """
        if self._thinking_draft_id is None:
            return None
        try:
            payload = RichMessageBuilder.build_thinking_draft(
                thoughts="\u200b",
                chat_id=self._chat_id,
                thread_id=self._thread_id or None,
            )
            payload["draft_id"] = self._thinking_draft_id
            bot = await _get_scheduler_bot()
            await bot.do_api_request("sendRichMessageDraft", api_kwargs=payload)
        except Exception as e:
            logger.bind(user_id=self._user_id).warning(f"Failed to clear thinking draft for {self.chat_data}: {e}.")
        finally:
            self._thinking_draft_id = None
        return None

    async def send_llm_thoughts(self, thoughts: str) -> None:
        """Send LLM thoughts as a native Telegram ``<tg-thinking>`` draft.

        Mirrors ``TelegramInterface.send_llm_thoughts``: the reasoning is
        delivered through the shared scheduler Bot as an ephemeral
        ``sendRichMessageDraft`` and cleared before the final message (see
        ``send_message``). Deliberately NO plain-text fallback: LLM thoughts
        must never become chat content in scheduled turns, so a delivery
        failure is only logged and the draft is simply not shown.

        Args:
            thoughts: The LLM reasoning text to display.
        """
        if not thoughts or thoughts == "No content":
            return None

        if self._thinking_draft_id is None:
            self._thinking_draft_id = random.randint(1, 2**31 - 1)

        payload = RichMessageBuilder.build_thinking_draft(
            thoughts=thoughts,
            chat_id=self._chat_id,
            thread_id=self._thread_id or None,
        )
        payload["draft_id"] = self._thinking_draft_id

        try:
            bot = await _get_scheduler_bot()
            await bot.do_api_request("sendRichMessageDraft", api_kwargs=payload)
        except Exception as e:
            logger.bind(user_id=self._user_id).warning(
                f"Failed to send LLM thoughts draft to {self.chat_data}: {e}. "
                f"Thoughts are never sent as plain text in scheduled turns."
            )
            # A failed draft is dropped (never re-sent as plain text); the id is
            # reset so a later thought chunk starts a fresh draft.
            self._thinking_draft_id = None
        return None

    async def send_message(self, message: str, reply: bool = True, **kwargs: Any) -> None:
        """Send a text message to the Telegram chat and thread fixed in the job context.

        The message is converted to MarkdownV2; if Telegram rejects it, it is
        re-sent in plain text chunks. Any active ``<tg-thinking>`` draft is
        cleared first so it does not overlap the final message.

        Args:
            message: The text content to send.
            reply: Whether to reply to the user's message; has no effect in
                scheduler context because there is no incoming message to
                reply to.
            **kwargs: Additional arguments for the message sending function.
        """
        await self._clear_thinking_draft()
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
        """Sends an audio file to the Telegram chat and thread fixed in the job context.

        Delivered through the shared scheduler Bot with HTML parse mode and
        audio-sized upload timeouts.

        Args:
            audio: The audio data or path to send.
            reply: Whether to reply to the user's message; has no effect in
                scheduler context because there is no incoming message to
                reply to.
            title: The title of the audio.
            caption: The caption for the audio.
            performer: The performer of the audio.
            duration: The duration of the audio in seconds.
            thumbnail: The thumbnail data for the audio.
            filename: The filename for the audio.
            **kwargs: Additional arguments for the audio sending function.
        """
        bot = await _get_scheduler_bot()
        await bot.send_audio(
            chat_id=self._chat_id,
            audio=audio,
            title=title,
            performer=performer,
            caption=caption,
            duration=duration,
            thumbnail=thumbnail,
            filename=filename,
            parse_mode="HTML",
            message_thread_id=self._thread_id or None,
            read_timeout=AUDIO_UPLOAD_TIMEOUT,
            write_timeout=AUDIO_UPLOAD_TIMEOUT,
        )

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
        """Sends a video file to the Telegram chat and thread fixed in the job context.

        Delivered through the shared scheduler Bot with HTML parse mode and
        file-sized upload timeouts.

        Args:
            video: The video data or path to send.
            reply: Whether to reply to the user's message; has no effect in
                scheduler context because there is no incoming message to
                reply to.
            title: The title of the video.
            caption: The caption for the video.
            duration: The duration of the video in seconds.
            thumbnail: The thumbnail data for the video.
            filename: The filename for the video.
            **kwargs: Additional arguments for the video sending function.
        """
        bot = await _get_scheduler_bot()
        await bot.send_video(
            chat_id=self._chat_id,
            video=video,
            caption=caption,
            duration=duration,
            thumbnail=thumbnail,
            filename=filename,
            message_thread_id=self._thread_id or None,
            parse_mode="HTML",
            read_timeout=FILE_UPLOAD_TIMEOUT,
            write_timeout=FILE_UPLOAD_TIMEOUT,
        )

    async def send_images(self, images: list[BytesIO] | list[str], reply: bool = True, **kwargs: Any) -> None:
        """Sends a list of images to the Telegram chat and thread fixed in the job context.

        URL lists are downloaded and delivered as a photo media group; if the
        media group fails, the original URLs are sent as a plain text message
        so the images are never silently lost. ``BytesIO`` buffers are
        partitioned by size: photos below ``PHOTOSIZE_UPLOAD`` go into a
        photos group, larger buffers below ``FILESIZE_UPLOAD`` are delivered
        as documents, and anything beyond the file size limit is skipped with
        an error log.

        Args:
            images: A list of image URLs or ``BytesIO`` buffers to send.
            reply: Whether to reply to the user's message; has no effect in
                scheduler context because there is no incoming message to
                reply to.
            **kwargs: Additional arguments for the image sending function.
        """
        if not images:
            logger.bind(user_id=self._user_id).warning(
                f"SchedulerInterface: send_images called with an empty list for {self.chat_data}; nothing to send."
            )
            return

        await self._send_chat_action(ChatAction.UPLOAD_PHOTO)
        bot = await _get_scheduler_bot()
        message_thread_id = self._thread_id or None

        if isinstance(images[0], str):
            logger.bind(user_id=self._user_id).info(
                f"Downloading {len(images)} images for {self.user_data} via URLs..."
            )
            image_files = [await download_image(image_url=str(url)) for url in images]
            try:
                logger.bind(user_id=self._user_id).info(
                    f"Uploading {len(images)} images to {self.user_data} in the {self.chat_data}"
                )
                await bot.send_media_group(
                    chat_id=self._chat_id,
                    media=[InputMediaPhoto(data) for data in image_files],
                    message_thread_id=message_thread_id,
                    read_timeout=IMAGE_UPLOAD_TIMEOUT,
                    write_timeout=IMAGE_UPLOAD_TIMEOUT,
                )
            except Exception as e:
                logger.bind(user_id=self._user_id).error(
                    f"{self.user_data} image generation request succeeded, but we couldn't send the image "
                    f"to {self.chat_data} due to exception: {e}. Trying to send it via text message..."
                )
                await self.send_message("\n".join(str(url) for url in images))
            return

        media_photos: list[BytesIO] = []
        media_docs: list[BytesIO] = []
        for file in images:
            if not isinstance(file, BytesIO):
                continue
            file.seek(0, 2)
            size = file.tell()
            file.seek(0)
            if size < FileSizeLimit.PHOTOSIZE_UPLOAD:
                media_photos.append(file)
            elif size < FileSizeLimit.FILESIZE_UPLOAD:
                media_docs.append(file)
            else:
                logger.bind(user_id=self._user_id).error(
                    f"{self.user_data} File size ({size}) exceeds file size limit, skipping it.."
                )
                continue

        if media_photos:
            await bot.send_media_group(
                chat_id=self._chat_id,
                media=[InputMediaPhoto(img) for img in media_photos],
                message_thread_id=message_thread_id,
                write_timeout=IMAGE_UPLOAD_TIMEOUT,
            )

        if media_docs:
            logger.bind(user_id=self._user_id).info(
                f"Uploading {len(media_docs)} image(s) as file(s) to {self.user_data} in the {self.chat_data}"
            )
            await bot.send_media_group(
                chat_id=self._chat_id,
                media=[InputMediaDocument(media=img, filename="file.jpeg") for img in media_docs],
                message_thread_id=message_thread_id,
                write_timeout=FILE_UPLOAD_TIMEOUT,
            )

    async def send_document(
        self,
        document: bytes | BytesIO,
        filename: str | None = None,
        caption: str | None = None,
        thumbnail: bytes | None = None,
        **kwargs: Any,
    ) -> None:
        """Sends a document file to the Telegram chat and thread fixed in the job context.

        Delivered through the shared scheduler Bot with file-sized upload
        timeouts.

        Args:
            document: The document data to send.
            filename: The filename for the document.
            caption: The caption for the document.
            thumbnail: The thumbnail data for the document.
            **kwargs: Additional arguments for the document sending function.
        """
        bot = await _get_scheduler_bot()
        await bot.send_document(
            chat_id=self._chat_id,
            document=document,
            filename=filename,
            caption=caption,
            thumbnail=thumbnail,
            message_thread_id=self._thread_id or None,
            read_timeout=FILE_UPLOAD_TIMEOUT,
            write_timeout=FILE_UPLOAD_TIMEOUT,
        )


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
