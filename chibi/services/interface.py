from abc import ABC
from io import BytesIO
from typing import Any

from telegram import Chat as TelegramChat
from telegram import File, Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from chibi.constants import AUDIO_UPLOAD_TIMEOUT, FILE_UPLOAD_TIMEOUT
from chibi.utils.telegram import send_answer_message, send_images


class UserInterface(ABC):
    @property
    def chat_id(self) -> str | int:
        raise NotImplementedError

    @property
    def user_id(self) -> int:
        raise NotImplementedError

    @property
    def user_data(self) -> str:
        raise NotImplementedError

    @property
    def chat_data(self) -> str:
        raise NotImplementedError

    async def get_text_prompt(self) -> str | None:
        raise NotImplementedError

    async def get_voice_prompt(self) -> BytesIO | None:
        raise NotImplementedError

    async def send_action_typing(self) -> None:
        raise NotImplementedError

    async def send_action_uploading_photo(self) -> None:
        raise NotImplementedError

    async def send_action_recording(self) -> None:
        raise NotImplementedError

    async def send_reaction(self, reaction: str) -> None:
        raise NotImplementedError

    async def delete_last_user_message(self) -> None:
        raise NotImplementedError

    async def send_message(self, message: str, reply: bool = True, **kwargs: Any) -> None:
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
        raise NotImplementedError

    async def send_images(self, images: list[BytesIO] | list[str], reply: bool = True, **kwargs: Any) -> None:
        raise NotImplementedError

    async def send_document(
        self,
        document: bytes | BytesIO,
        filename: str | None = None,
        caption: str | None = None,
        thumbnail: bytes | None = None,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError


class TelegramInterface(UserInterface):
    def __init__(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        self.update = update
        self.context = context

    @property
    def _chat(self) -> TelegramChat:
        if chat := self.update.effective_chat:
            return chat
        raise ValueError("Telegram incoming update does not contain valid chat data.")

    @property
    def chat_id(self) -> str | int:
        return self._chat.id

    @property
    def user_id(self) -> int:
        if user := self.update.effective_user:
            return user.id
        raise ValueError("Telegram incoming update does not contain valid user data.")

    @property
    def user_data(self) -> str:
        if user := self.update.effective_user:
            return f"{user.name} ({user.id})"
        raise ValueError("Telegram incoming update does not contain valid user data.")

    @property
    def chat_data(self) -> str:
        return f"{self._chat.type.upper()} chat ({self._chat.id})"

    async def get_text_prompt(self) -> str | None:
        if message := self.update.message:
            return message.text
        raise ValueError("Telegram incoming update does not contain valid message data.")

    async def get_voice_prompt(self) -> BytesIO | None:
        if not self.update.message:
            raise ValueError("Telegram incoming update does not contain valid message data.")
        if voice := self.update.message.voice:
            file_id = voice.file_id
            file: File = await self.context.bot.get_file(file_id)
            voice_prompt = BytesIO()
            await file.download_to_memory(out=voice_prompt)
            voice_prompt.seek(0)
            return voice_prompt
        return None

    async def send_action_typing(self) -> None:
        await self._chat.send_chat_action(action=ChatAction.TYPING)

    async def send_action_uploading_photo(self) -> None:
        await self._chat.send_chat_action(action=ChatAction.UPLOAD_PHOTO)

    async def send_action_recording(self) -> None:
        await self._chat.send_chat_action(action=ChatAction.RECORD_VOICE)

    async def send_reaction(self, reaction: str) -> None:
        if message := self.update.message:
            await message.set_reaction(reaction=reaction, is_big=True)
        raise ValueError("Telegram incoming update does not contain valid message data.")

    async def delete_last_user_message(self) -> None:
        if message := self.update.effective_message:
            try:
                await message.delete()
            except Exception:
                pass
        return None

    async def send_message(self, message: str, reply: bool = True, **kwargs: Any) -> None:
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
        await self.context.bot.send_audio(
            chat_id=self.chat_id,
            audio=audio,
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
        await send_images(images=images, update=self.update, context=self.context)

    async def send_document(
        self,
        document: bytes | BytesIO,
        filename: str | None = None,
        caption: str | None = None,
        thumbnail: bytes | None = None,
        **kwargs: Any,
    ) -> None:
        await self.context.bot.send_document(
            chat_id=self.chat_id,
            document=document,
            filename=filename,
            caption=caption,
            thumbnail=thumbnail,
        )
