from abc import ABC
from io import BytesIO
from typing import Any

from loguru import logger
from telegram import Chat as TelegramChat
from telegram import File, Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from chibi.constants import AUDIO_UPLOAD_TIMEOUT, FILE_UPLOAD_TIMEOUT
from chibi.utils.telegram import send_answer_message, send_images


class UserInterface(ABC):
    @property
    def chat_id(self) -> str | int:
        """Returns the unique identifier for the current chat.

        Returns:
            The chat identifier.
        """
        raise NotImplementedError

    @property
    def user_id(self) -> int:
        """Returns the unique identifier for the current user.

        Returns:
            The user identifier.
        """
        raise NotImplementedError

    @property
    def user_data(self) -> str:
        """Returns a string representation of the user data.

        Returns:
            The user data string.
        """
        raise NotImplementedError

    @property
    def chat_data(self) -> str:
        """Returns a string representation of the chat data.

        Returns:
            The chat data string.
        """
        raise NotImplementedError

    @property
    def attached_document(self) -> dict[str, str] | None:
        """Returns the attached document data if present, otherwise None.

        Returns:
            The document data dictionary or None.
        """
        raise NotImplementedError

    @property
    def attached_document_caption(self) -> str | None:
        """Returns the caption of the attached document if present, otherwise None.

        Returns:
            The caption string or None.
        """
        raise NotImplementedError

    async def get_text_prompt(self) -> str | None:
        """Retrieves the text prompt from the current message.

        Returns:
            The text prompt string or None.
        """
        raise NotImplementedError

    async def get_voice_prompt(self) -> BytesIO | None:
        """Retrieves the voice prompt as a BytesIO object if present, otherwise None.

        Returns:
            The voice prompt BytesIO object or None.
        """
        raise NotImplementedError

    async def get_caption(self) -> str | None:
        raise NotImplementedError

    def set_caption(self, caption: str) -> None:
        raise NotImplementedError

    async def send_action_typing(self) -> None:
        """Sends a typing action to the user."""
        raise NotImplementedError

    async def send_action_uploading_photo(self) -> None:
        """Sends an uploading photo action to the user."""
        raise NotImplementedError

    async def send_action_recording(self) -> None:
        """Sends a recording voice action to the user."""
        raise NotImplementedError

    async def send_reaction(self, reaction: str) -> None:
        """Sends a reaction to the user's message.

        Args:
            reaction: The reaction to send.
        """
        raise NotImplementedError

    async def delete_last_user_message(self) -> None:
        """Deletes the last message sent by the user."""
        raise NotImplementedError

    async def send_message(self, message: str, reply: bool = True, **kwargs: Any) -> None:
        """Sends a text message to the user.

        Args:
            message: The text content to send.
            reply: Whether to reply to the user's message.
            **kwargs: Additional arguments for the message sending function.
        """
        raise NotImplementedError

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
        """
        raise NotImplementedError

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
        """
        raise NotImplementedError

    async def send_images(self, images: list[BytesIO] | list[str], reply: bool = True, **kwargs: Any) -> None:
        """Sends a list of images to the user.

        Args:
            images: A list of image data or paths to send.
            reply: Whether to reply to the user's message.
            **kwargs: Additional arguments for the image sending function.
        """
        raise NotImplementedError

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
        """
        raise NotImplementedError


class TelegramInterface(UserInterface):
    def __init__(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        self.update = update
        self.context = context
        self._caption: str | None = None

    @property
    def _chat(self) -> TelegramChat:
        """Internal helper to access the current Telegram chat.

        Returns:
            The Telegram chat object.
        """
        if chat := self.update.effective_chat:
            return chat
        raise ValueError("Telegram incoming update does not contain valid chat data.")

    @property
    def chat_id(self) -> str | int:
        """Returns the unique identifier for the current Telegram chat.

        Returns:
            The chat identifier.
        """
        return self._chat.id

    @property
    def user_id(self) -> int:
        """Returns the unique identifier for the current Telegram user.

        Returns:
            The user identifier.
        """
        if user := self.update.effective_user:
            return user.id
        raise ValueError("Telegram incoming update does not contain valid user data.")

    @property
    def user_data(self) -> str:
        """Returns a string representation of the current Telegram user.

        Returns:
            The user data string.
        """
        if user := self.update.effective_user:
            return f"{user.name} ({user.id})"
        raise ValueError("Telegram incoming update does not contain valid user data.")

    @property
    def chat_data(self) -> str:
        """Returns a string representation of the current Telegram chat.

        Returns:
            The chat data string.
        """
        return f"{self._chat.type.upper()} chat ({self._chat.id})"

    @property
    def attached_document(self) -> dict[str, str] | None:
        """Returns the attached document or photo data from the message if present.

        Returns:
            The document data dictionary or None.
        """
        message = self.update.effective_message
        if not message:
            return None
        if document := message.document:
            return document.to_dict()
        if not message.photo:
            return None
        photo = message.photo[-1]
        return photo.to_dict()

    @property
    def attached_document_caption(self) -> str | None:
        """Returns the caption of the attached document or photo if present.

        Returns:
            The caption string or None.
        """
        message = self.update.effective_message
        if not message:
            return None
        return message.caption

    async def get_text_prompt(self) -> str | None:
        """Retrieves the text or caption from the current Telegram message.

        Returns:
            The text prompt string or None.
        """
        if message := self.update.effective_message:
            return message.text or message.caption
        raise ValueError("Telegram incoming update does not contain valid message data.")

    async def get_voice_prompt(self) -> BytesIO | None:
        """Downloads and returns the voice message from the current Telegram message as a BytesIO object.

        Returns:
            The voice prompt BytesIO object or None.
        """
        if not self.update.effective_message:
            return None
        if voice := self.update.effective_message.voice:
            file_id = voice.file_id
            file: File = await self.context.bot.get_file(file_id)
            voice_prompt = BytesIO()
            await file.download_to_memory(out=voice_prompt)
            voice_prompt.seek(0)
            return voice_prompt
        return None

    async def send_action_typing(self) -> None:
        """Sends a typing action to the Telegram chat."""
        await self._chat.send_chat_action(action=ChatAction.TYPING)

    async def send_action_uploading_photo(self) -> None:
        """Sends an uploading photo action to the Telegram chat."""
        await self._chat.send_chat_action(action=ChatAction.UPLOAD_PHOTO)

    async def send_action_recording(self) -> None:
        """Sends a recording voice action to the Telegram chat."""
        await self._chat.send_chat_action(action=ChatAction.RECORD_VOICE)

    async def send_reaction(self, reaction: str) -> None:
        """Sends a reaction to the user's message in Telegram.

        Args:
            reaction: The reaction to send.
        """
        if message := self.update.effective_message:
            await message.set_reaction(reaction=reaction, is_big=True)
            return None
        logger.warning("We tried to set the reaction on user message, but no user message found in TG update.")

    async def delete_last_user_message(self) -> None:
        """Deletes the last message sent by the user in Telegram."""
        if message := self.update.effective_message:
            try:
                await message.delete()
            except Exception as e:
                logger.error(f"Error deleting last user message: {e}")
                pass
        return None

    async def send_message(self, message: str, reply: bool = True, **kwargs: Any) -> None:
        """Sends a text message to the Telegram chat.

        Args:
            message: The text content to send.
            reply: Whether to reply to the user's message.
            **kwargs: Additional arguments for the message sending function.
        """
        await send_answer_message(message=message, update=self.update, context=self.context, reply=reply)

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
        """Sends an audio file to the Telegram chat.

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
        """
        await self.context.bot.send_audio(
            chat_id=self.chat_id,
            audio=audio,
            title=title,
            performer=performer,
            caption=caption,
            duration=duration,
            thumbnail=thumbnail,
            filename=filename,
            parse_mode="HTML",
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
        """Sends a video file to the Telegram chat.

        Args:
            video: The video data or path to send.
            reply: Whether to reply to the user's message.
            title: The title of the video.
            caption: The caption for the video.
            duration: The duration of the video in seconds.
            thumbnail: The thumbnail data for the video.
            filename: The filename for the video.
            **kwargs: Additional arguments for the video sending function.
        """
        await self.context.bot.send_video(
            chat_id=self.chat_id,
            video=video,
            caption=caption,
            duration=duration,
            thumbnail=thumbnail,
            filename=filename,
            parse_mode="HTML",
            read_timeout=FILE_UPLOAD_TIMEOUT,
            write_timeout=FILE_UPLOAD_TIMEOUT,
        )

    async def send_images(self, images: list[BytesIO] | list[str], reply: bool = True, **kwargs: Any) -> None:
        """Sends a list of images to the Telegram chat.

        Args:
            images: A list of image data or paths to send.
            reply: Whether to reply to the user's message.
            **kwargs: Additional arguments for the image sending function.
        """
        await send_images(images=images, update=self.update, context=self.context)

    async def send_document(
        self,
        document: bytes | BytesIO,
        filename: str | None = None,
        caption: str | None = None,
        thumbnail: bytes | None = None,
        **kwargs: Any,
    ) -> None:
        """Sends a document file to the Telegram chat.

        Args:
            document: The document data to send.
            filename: The filename for the document.
            caption: The caption for the document.
            thumbnail: The thumbnail data for the document.
            **kwargs: Additional arguments for the document sending function.
        """
        await self.context.bot.send_document(
            chat_id=self.chat_id,
            document=document,
            filename=filename,
            caption=caption,
            thumbnail=thumbnail,
        )

    async def get_caption(self) -> str | None:
        if self._caption:
            return self._caption

        if message := self.update.effective_message:
            return message.caption

        return None

    def set_caption(self, caption: str) -> None:
        self._caption = caption
        return None
