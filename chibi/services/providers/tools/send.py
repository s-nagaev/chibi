import re
from typing import Any, Unpack

import httpx
from loguru import logger
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition
from telegram import InputMediaAudio, InputMediaPhoto, InputMediaVideo

from chibi.constants import AUDIO_UPLOAD_TIMEOUT, FILE_UPLOAD_TIMEOUT, IMAGE_UPLOAD_TIMEOUT
from chibi.services.providers.tools.exceptions import ToolException
from chibi.services.providers.tools.tool import ChibiTool
from chibi.services.providers.tools.utils import AdditionalOptions, download
from chibi.utils.telegram import get_telegram_chat


class SendTextFileTool(ChibiTool):
    register = True
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="send_text_based_file",
            description="Send a data as a text-based file (.md, .txt, .rst, .py, etc)  to user.",
            parameters={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "File content"},
                    "filename": {"type": "string", "description": "File name including extension, i.e. 'info.txt'"},
                },
                "required": ["content", "filename"],
            },
        ),
    )
    name = "send_text_based_file"

    @classmethod
    async def function(cls, content: str, filename: str, **kwargs: Unpack[AdditionalOptions]) -> dict[str, str]:
        user_id = kwargs.get("user_id")
        if not user_id:
            raise ToolException("This function requires user_id to be automatically provided.")

        telegram_context = kwargs.get("telegram_context")
        telegram_update = kwargs.get("telegram_update")

        if telegram_context is None or telegram_update is None:
            raise ToolException(
                "This function requires telegram context & telegram update to be automatically provided."
            )

        from chibi.utils.telegram import send_text_file

        await send_text_file(file_content=content, file_name=filename, update=telegram_update, context=telegram_context)

        return {"detail": "File was successfully sent."}


class SendAudioTool(ChibiTool):
    register = True
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="send_audio",
            description="Send an audio file to the user in Telegram.",
            parameters={
                "type": "object",
                "properties": {
                    "audio_url": {
                        "type": "string",
                        "description": (
                            "URL to the audio file (MP3, OGG, etc.). Telegram will download it automatically."
                        ),
                    },
                    "title": {
                        "type": "string",
                        "description": "Audio title/track name.",
                    },
                    "performer": {
                        "type": "string",
                        "description": "Performer/artist name (optional).",
                    },
                    "duration": {
                        "type": "integer",
                        "description": "Audio duration in seconds (optional).",
                    },
                    "thumbnail_url": {
                        "type": "string",
                        "description": "URL to thumbnail image (optional). Will be downloaded and attached.",
                    },
                    "caption": {
                        "type": "string",
                        "description": "Caption text to display with the audio (optional).",
                    },
                },
                "required": ["audio_url"],
            },
        ),
    )
    name = "send_audio"

    @classmethod
    async def function(
        cls,
        audio_url: str,
        title: str | None = None,
        performer: str | None = None,
        duration: int | None = None,
        thumbnail_url: str | None = None,
        caption: str | None = None,
        **kwargs: Unpack[AdditionalOptions],
    ) -> dict[str, str]:
        telegram_context = kwargs.get("telegram_context")
        telegram_update = kwargs.get("telegram_update")

        if telegram_context is None or telegram_update is None:
            raise ToolException(
                "This function requires telegram context & telegram update to be automatically provided."
            )

        logger.log("TOOL", f"Sending audio to user: {audio_url}")

        # Download thumbnail if provided
        thumbnail_data: bytes | None = None
        audio_data: bytes | None = None
        if audio_url.startswith("http"):
            audio_data = await download(url=audio_url)

        if thumbnail_url:
            thumbnail_data = await download(url=thumbnail_url)

        filename = None
        if title:
            clean_title = re.sub(r"[^\w\s-]", "", title).strip()
            clean_title = re.sub(r"[-\s]+", "_", clean_title)
            filename = f"{clean_title[:50]}.mp3"

        await telegram_context.bot.send_audio(
            chat_id=get_telegram_chat(update=telegram_update).id,
            audio=audio_data or audio_url,
            title=title,
            performer=performer,
            duration=duration,
            thumbnail=thumbnail_data,
            caption=caption,
            filename=filename,
            parse_mode="HTML",
            read_timeout=AUDIO_UPLOAD_TIMEOUT,
            write_timeout=AUDIO_UPLOAD_TIMEOUT,
        )

        return {"detail": "Audio was successfully sent."}


class SendVideoTool(ChibiTool):
    register = True
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="send_video",
            description="Send a video file to the user in Telegram.",
            parameters={
                "type": "object",
                "properties": {
                    "video_url": {
                        "type": "string",
                        "description": "URL to the video file (MP4, etc.). Telegram will download it automatically.",
                    },
                    "caption": {
                        "type": "string",
                        "description": "Caption text to display with the video (optional).",
                    },
                    "duration": {
                        "type": "integer",
                        "description": "Video duration in seconds (optional).",
                    },
                    "width": {
                        "type": "integer",
                        "description": "Video width in pixels (optional).",
                    },
                    "height": {
                        "type": "integer",
                        "description": "Video height in pixels (optional).",
                    },
                    "thumbnail_url": {
                        "type": "string",
                        "description": "URL to thumbnail image (optional). Will be downloaded and attached.",
                    },
                },
                "required": ["video_url"],
            },
        ),
    )
    name = "send_video"

    @classmethod
    async def function(
        cls,
        video_url: str,
        caption: str | None = None,
        duration: int | None = None,
        width: int | None = None,
        height: int | None = None,
        thumbnail_url: str | None = None,
        **kwargs: Unpack[AdditionalOptions],
    ) -> dict[str, str]:
        telegram_context = kwargs.get("telegram_context")
        telegram_update = kwargs.get("telegram_update")

        if telegram_context is None or telegram_update is None:
            raise ToolException(
                "This function requires telegram context & telegram update to be automatically provided."
            )

        logger.log("TOOL", f"Sending video to user: {video_url}")

        # Download thumbnail if provided
        thumbnail_bytes = None
        if thumbnail_url:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(thumbnail_url, timeout=30.0)
                    response.raise_for_status()
                    thumbnail_bytes = response.content
                    logger.log("TOOL", f"Downloaded thumbnail: {len(thumbnail_bytes)} bytes")
            except Exception as e:
                logger.warning(f"Failed to download thumbnail: {e}")

        # Send video
        await telegram_context.bot.send_video(
            chat_id=get_telegram_chat(update=telegram_update).id,
            video=video_url,
            caption=caption,
            duration=duration,
            width=width,
            height=height,
            thumbnail=thumbnail_bytes,
            parse_mode="HTML",
            read_timeout=60,
            write_timeout=60,
        )

        return {"detail": "Video was successfully sent."}


class SendImageTool(ChibiTool):
    register = True
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="send_image",
            description="Send an image (photo) to the user in Telegram.",
            parameters={
                "type": "object",
                "properties": {
                    "image_url": {
                        "type": "string",
                        "description": (
                            "URL to the image file (JPEG, PNG, etc.). Telegram will download it automatically."
                        ),
                    },
                    "caption": {
                        "type": "string",
                        "description": "Caption text to display with the image (optional).",
                    },
                },
                "required": ["image_url"],
            },
        ),
    )
    name = "send_image"

    @classmethod
    async def function(
        cls,
        image_url: str,
        caption: str | None = None,
        **kwargs: Unpack[AdditionalOptions],
    ) -> dict[str, str]:
        telegram_context = kwargs.get("telegram_context")
        telegram_update = kwargs.get("telegram_update")

        if telegram_context is None or telegram_update is None:
            raise ToolException(
                "This function requires telegram context & telegram update to be automatically provided."
            )

        logger.log("TOOL", f"Sending image to user: {image_url}")

        # Send photo
        await telegram_context.bot.send_photo(
            chat_id=get_telegram_chat(update=telegram_update).id,
            photo=image_url,
            caption=caption,
            parse_mode="HTML",
            read_timeout=IMAGE_UPLOAD_TIMEOUT,
            write_timeout=IMAGE_UPLOAD_TIMEOUT,
        )

        return {"detail": "Image was successfully sent."}


class SendMediaGroupTool(ChibiTool):
    register = True
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="send_media_group",
            description=(
                "Send a group of media files (2-10 items) to the user in Telegram as an album. "
                "Use this when you need to send multiple related images, videos, or audio files together. "
                "For a single media file, use send_image, send_video, or send_audio instead."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "media": {
                        "type": "array",
                        "description": "Array of media items to send as a group (album).",
                        "minItems": 2,
                        "maxItems": 10,
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["photo", "video", "audio"],
                                    "description": "Type of media: photo, video, or audio.",
                                },
                                "url": {
                                    "type": "string",
                                    "description": "URL to the media file. Telegram will download it automatically.",
                                },
                                "caption": {
                                    "type": "string",
                                    "description": "Caption for this specific media item (optional).",
                                },
                                "thumbnail_url": {
                                    "type": "string",
                                    "description": "URL to thumbnail image for video/audio (optional).",
                                },
                            },
                            "required": ["type", "url"],
                        },
                    },
                },
                "required": ["media"],
            },
        ),
    )
    name = "send_media_group"

    @classmethod
    async def function(
        cls,
        media: list[dict[str, Any]],
        **kwargs: Unpack[AdditionalOptions],
    ) -> dict[str, str]:
        telegram_context = kwargs.get("telegram_context")
        telegram_update = kwargs.get("telegram_update")

        if telegram_context is None or telegram_update is None:
            raise ToolException(
                "This function requires telegram context & telegram update to be automatically provided."
            )

        if not media or len(media) < 2:
            raise ToolException("Media group must contain at least 2 items.")

        if len(media) > 10:
            raise ToolException("Media group cannot contain more than 10 items (Telegram limit).")

        logger.log("TOOL", f"Sending media group with {len(media)} items to user")

        # Build media group
        media_group: list[InputMediaPhoto | InputMediaVideo | InputMediaAudio] = []

        for idx, item in enumerate(media):
            media_type = item.get("type")
            url = item.get("url")
            caption = item.get("caption")
            thumbnail_url = item.get("thumbnail_url")

            if not media_type or not url:
                raise ToolException(f"Media item {idx} is missing required 'type' or 'url' field.")

            # Download thumbnail if provided (for video/audio)
            thumbnail_bytes = None
            if thumbnail_url and media_type in ["video", "audio"]:
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(thumbnail_url, timeout=30.0)
                        response.raise_for_status()
                        thumbnail_bytes = response.content
                        logger.log("TOOL", f"Downloaded thumbnail for item {idx}: {len(thumbnail_bytes)} bytes")
                except Exception as e:
                    logger.warning(f"Failed to download thumbnail for item {idx}: {e}")

            # Create appropriate InputMedia object
            if media_type == "photo":
                media_group.append(
                    InputMediaPhoto(
                        media=url,
                        caption=caption,
                        parse_mode="HTML",
                    )
                )
            elif media_type == "video":
                media_group.append(
                    InputMediaVideo(
                        media=url,
                        caption=caption,
                        thumbnail=thumbnail_bytes,
                        parse_mode="HTML",
                    )
                )
            elif media_type == "audio":
                media_group.append(
                    InputMediaAudio(
                        media=url,
                        caption=caption,
                        thumbnail=thumbnail_bytes,
                        parse_mode="HTML",
                    )
                )
            else:
                raise ToolException(f"Invalid media type '{media_type}' for item {idx}. Must be: photo, video, audio.")

        # Send media group
        await telegram_context.bot.send_media_group(
            chat_id=get_telegram_chat(update=telegram_update).id,
            media=media_group,
            read_timeout=FILE_UPLOAD_TIMEOUT,
            write_timeout=FILE_UPLOAD_TIMEOUT,
        )

        return {"detail": f"Media group with {len(media)} items was successfully sent."}
