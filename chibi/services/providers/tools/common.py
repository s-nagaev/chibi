import re
from asyncio import sleep
from typing import TYPE_CHECKING, Any, Optional, Unpack

import httpx
from loguru import logger
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition
from telegram import InputMediaAudio, InputMediaPhoto, InputMediaVideo, Update
from telegram.ext import ContextTypes

from chibi.config import gpt_settings, telegram_settings
from chibi.constants import AUDIO_UPLOAD_TIMEOUT
from chibi.schemas.app import ChatResponseSchema, ModelChangeSchema
from chibi.services.providers.constants.suno import POLLING_ATTEMPTS_WAIT_BETWEEN
from chibi.services.providers.tools.exceptions import ToolException
from chibi.services.providers.tools.tool import ChibiTool
from chibi.services.providers.tools.utils import AdditionalOptions, download, get_sub_agent_response
from chibi.services.task_manager import task_manager
from chibi.services.user import generate_image, user_has_reached_images_generation_limit
from chibi.utils.telegram import get_telegram_chat

if TYPE_CHECKING:
    from chibi.services.providers import Suno


class TextToSpeechTool(ChibiTool):
    register = bool(gpt_settings.openai_key)
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="text_to_speech",
            description=(
                "Send an audio file with speech to user. Use it when user ask you or sending you voice messages."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to speech"},
                },
                "required": ["text"],
            },
        ),
    )
    name = "text_to_speech"

    @classmethod
    async def function(cls, text: str, **kwargs: Unpack[AdditionalOptions]) -> dict[str, str]:
        from chibi.services.providers import OpenAI, RegisteredProviders
        from chibi.utils.telegram import send_audio

        telegram_context = kwargs.get("telegram_context")
        telegram_update = kwargs.get("telegram_update")

        if telegram_context is None or telegram_update is None:
            raise ToolException(
                "This function requires telegram context & telegram update to be automatically provided."
            )
        logger.log("TOOL", "Sending voice message to user...")

        provider = RegisteredProviders().get(provider_name="OpenAI")
        if not isinstance(provider, OpenAI):
            raise ToolException("This function requires OpenAI provider.")  # TODO: temporary solution

        audio_data = await provider.speech(text=text)
        await send_audio(
            audio=audio_data,
            update=telegram_update,
            context=telegram_context,
        )
        return {"detail": "Audio was successfully sent."}


class GetAvailableImageModelsTool(ChibiTool):
    register = True
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="get_available_image_generation_models",
            description=("Get models and providers available for user for image generation."),
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    )
    name = "get_available_image_generation_models"

    @classmethod
    async def function(cls, **kwargs: Unpack[AdditionalOptions]) -> dict[str, Any]:
        user_id = kwargs.get("user_id")
        if not user_id:
            raise ToolException("This function requires user_id to be automatically provided.")

        logger.log("TOOL", f"Getting available image generation models for user {user_id}...")

        from chibi.services.user import get_models_available

        data: list[ModelChangeSchema] = await get_models_available(user_id=user_id, image_generation=True)

        return {
            "available_models": [info.model_dump(include={"provider", "name", "display_name"}) for info in data],
        }


class GetAvailableLLMModelsTool(ChibiTool):
    register = True
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="get_available_llm_models",
            description="Get LLM models and providers available for user.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    )
    name = "get_available_llm_models"

    @classmethod
    async def function(cls, **kwargs: Unpack[AdditionalOptions]) -> dict[str, Any]:
        user_id = kwargs.get("user_id")
        if not user_id:
            raise ToolException("This function requires user_id to be automatically provided.")

        logger.log("TOOL", f"Getting available LLM models for user {user_id}...")

        from chibi.services.user import get_models_available

        data: list[ModelChangeSchema] = await get_models_available(user_id=user_id, image_generation=False)

        return {
            "available_models": [info.model_dump(include={"provider", "name", "display_name"}) for info in data],
        }


class GenerateImageTool(ChibiTool):
    register = True
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="generate_image",
            description=(
                "Generate image using one of the available models. You won’t see the image itself, only a message "
                "about whether the operation was successful or not. Check available providers and models first. "
                "Use your knowledge to adapt the prompt for a specific model to achieve the best result. "
                f"The aspect ratio ({gpt_settings.image_aspect_ratio}), size, and image quality are set globally "
                f"and cannot be changed via the prompt"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "provider": {"type": "string", "description": "Provider name, i.e. 'Gemini'"},
                    "image_model": {"type": "string", "description": "Model name, i.e. 'dall-e-3'"},
                    "prompt": {"type": "string", "description": "Image generation prompt. English recommended"},
                    "execute_in_background": {
                        "type": "boolean",
                        "description": "Execute image generation in background.",
                        "default": True,
                    },
                },
                "required": ["provider", "image_model", "prompt"],
            },
        ),
    )
    name = "generate_image"

    @classmethod
    async def generate_and_send_image(
        cls,
        user_id: int,
        provider: str,
        model: str,
        prompt: str,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        from chibi.utils.telegram import send_images

        images = await generate_image(user_id=user_id, provider_name=provider, model=model, prompt=prompt)
        await send_images(images=images, update=update, context=context)
        return None

    @classmethod
    async def function(
        cls,
        provider: str,
        image_model: str,
        prompt: str,
        execute_in_background: bool = True,
        **kwargs: Unpack[AdditionalOptions],
    ) -> dict[str, str]:
        user_id = kwargs.get("user_id")
        if not user_id:
            raise ToolException("This function requires user_id to be automatically provided.")
        telegram_context = kwargs.get("telegram_context")
        telegram_update = kwargs.get("telegram_update")

        if telegram_context is None or telegram_update is None:
            raise ToolException(
                "This function requires telegram context & telegram update to be automatically provided."
            )

        if await user_has_reached_images_generation_limit(user_id=user_id):
            raise ToolException("User has reached image generation monthly limit.")

        coro = cls.generate_and_send_image(
            user_id=user_id,
            provider=provider,
            model=image_model,
            prompt=prompt,
            update=telegram_update,
            context=telegram_context,
        )
        if execute_in_background:
            task_manager.run_task(coro)
            return {"detail": "Image generation was successfully scheduled. User will receive it soon."}

        await coro
        return {"detail": "Image was successfully generated and sent."}


class DelegateTool(ChibiTool):
    register = True
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="delegate_task",
            description=(
                "Delegate exactly one task to a sub-agent - an LLM identical to you. The prompt should be "
                "exhaustive and expect a concrete result, or an explanation for its absence. The task should be "
                "as atomic as possible. Delegate preferably tasks that involve processing large volumes of "
                "information, to avoid saturating your context. If no model/provider specified, your model will be used"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Prompt"},
                    "provider_name": {"type": "string", "description": "Provider name, i.e. 'OpenAI'"},
                    "model_name": {"type": "string", "description": "Model name, i.e. 'gpt-5.2'"},
                },
                "required": ["prompt"],
            },
        ),
    )
    name = "delegate_task"

    @classmethod
    async def function(
        cls,
        prompt: str,
        provider_name: str | None = None,
        model_name: str | None = None,
        **kwargs: Unpack[AdditionalOptions],
    ) -> dict[str, str]:
        user_id = kwargs.get("user_id")
        if not user_id:
            raise ToolException("This function requires user_id to be automatically provided.")
        model = kwargs.get("model")
        if not model:
            raise ToolException("This function requires model to be automatically provided.")
        logger.log("DELEGATE", f"[{model}] Delegating a task to {model_name or model}: {prompt}")

        response: ChatResponseSchema = await get_sub_agent_response(
            user_id=user_id, prompt=prompt, provider_name=provider_name, model_name=model_name
        )
        logger.log("SUBAGENT", f"[{model_name or model}] Delegated task is done: {response.answer}")

        return {"response": response.answer}


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
            read_timeout=60,
            write_timeout=60,
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
            read_timeout=60,
            write_timeout=60,
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
            read_timeout=60,
            write_timeout=60,
        )

        return {"detail": f"Media group with {len(media)} items was successfully sent."}


class GenerateMusicViaSunoTool(ChibiTool):
    register = bool(gpt_settings.suno_key)
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="generate_music_via_suno",
            description="Generate music via Suno AI using unofficial API.",
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Description of the music with or without lyrics. 500 characters maximum",
                    },
                    "instrumental": {
                        "type": "boolean",
                        "description": "Whether to generate instrumental only music.",
                        "default": False,
                    },
                    "suno_model": {
                        "type": "string",
                        "description": "Model version. Available options: V4, V4_5, V4_5PLUS, V4_5ALL, V5",
                        "default": "V5",
                    },
                    "execute_in_background": {
                        "type": "boolean",
                        "description": "Execute image generation in background.",
                        "default": True,
                    },
                },
                "required": ["prompt"],
            },
        ),
    )
    name = "generate_music_via_suno"
    _provider: Optional["Suno"] = None

    @classmethod
    async def function(
        cls,
        prompt: str,
        suno_model: str = "V5",
        instrumental: bool = False,
        execute_in_background: bool = True,
        **kwargs: Unpack[AdditionalOptions],
    ) -> dict[str, Any]:
        telegram_context = kwargs.get("telegram_context")
        telegram_update = kwargs.get("telegram_update")

        if telegram_context is None or telegram_update is None:
            raise ToolException(
                "This function requires telegram context & telegram update to be automatically provided."
            )
        logger.log("TOOL", f"Generating music via Suno. Prompt: {prompt}")

        task_id = await cls._get_provider().order_music_generation(
            prompt=prompt, instrumental_only=instrumental, model=suno_model
        )

        logger.log("TOOL", f"[Task #{task_id}] Music generation request accepted by API.")

        coro = cls._poll_and_send_audio(
            task_id=task_id, telegram_context=telegram_context, telegram_update=telegram_update
        )

        if execute_in_background:
            logger.log("TOOL", f"[Task #{task_id}] Polling the generation result in the background...")
            task_manager.run_task(coro)
            return {
                "detail": "Music generation was successfully scheduled. User will receive it soon.",
                "suno_task_id": task_id,
            }

        return await coro

    @classmethod
    def _get_provider(cls) -> "Suno":
        from chibi.services.providers import RegisteredProviders, Suno

        if cls._provider:
            return cls._provider
        provider = RegisteredProviders().get(provider_name="Suno")
        if not isinstance(provider, Suno):
            raise ToolException("This function requires Suno provider to be set.")
        cls._provider = provider
        return provider

    @classmethod
    async def _poll_and_send_audio(
        cls,
        task_id: int | str,
        telegram_context: ContextTypes.DEFAULT_TYPE,
        telegram_update: Update,
    ) -> dict[str, Any]:
        await sleep(POLLING_ATTEMPTS_WAIT_BETWEEN)
        music_generation_result = await cls._get_provider().poll_result(task_id=task_id)

        if not music_generation_result.data:
            raise ToolException(
                f"SunoGetGenerationDetails does not contain data: {music_generation_result.model_dump()}"
            )

        if not music_generation_result.data.response:
            raise ToolException(
                f"SunoGetGenerationDetails.data does not contain response: {music_generation_result.model_dump()}"
            )

        if not music_generation_result.data.response.suno_data:
            raise ToolException(
                f"SunoGetGenerationDetails.data.response does not contain suno_data: "
                f"{music_generation_result.model_dump()}"
            )

        for version, suno_data in enumerate(music_generation_result.data.response.suno_data, start=1):
            music_url = (
                suno_data.audio_url
                or suno_data.source_audio_url
                or suno_data.stream_audio_url
                or suno_data.source_stream_audio_url
            )
            if not music_url:
                logger.warning(f"Suno task #{task_id} does not contain audio URL. Suno data ID: {suno_data.id}")
                continue

            image_url = suno_data.image_url or suno_data.source_image_url
            title = f"{suno_data.title} v{version}"
            chat_id = get_telegram_chat(update=telegram_update).id

            logger.log("TOOL", f"[Suno] Audio and thumbnail downloaded. Sending it to the chat #{chat_id}...")
            await telegram_context.bot.send_audio(
                chat_id=chat_id,
                audio=await download(str(music_url)) or str(music_url),
                title=title,
                performer=f"{telegram_settings.bot_name} AI via Suno AI",
                duration=int(suno_data.duration) if suno_data.duration else None,
                thumbnail=await download(str(image_url)) if image_url else None,
                filename=f"{title.replace(' ', '_')}.mp3",
                parse_mode="HTML",
                read_timeout=AUDIO_UPLOAD_TIMEOUT,
                write_timeout=AUDIO_UPLOAD_TIMEOUT,
            )
        return {
            "detail": "Music was successfully generated and sent to user",
            "suno_task_data": music_generation_result.model_dump(),
        }


class GenerateAdvancedMusicViaSunoTool(GenerateMusicViaSunoTool):
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="generate_music_via_suno_custom_mode",
            description="Generate music via Suno AI using unofficial API. (Custom Mode)",
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": (
                            "Description of the music or lyrics. If contains lyrics, must contain"
                            "only lyrics. Limits: for V4 model - 3000 chars, for V4_5, V4_5PLUS, "
                            "V4_5ALL and V5 - 5000 chars."
                        ),
                    },
                    "style": {
                        "type": "string",
                        "description": (
                            "Music style, i.e. 'Uplifting trance, Techno, Eurodance'"
                            "Limits: for V4 model - 200 chars, for V4_5, V4_5PLUS, "
                            "V4_5ALL and V5 - 1000 chars"
                        ),
                    },
                    "title": {
                        "type": "string",
                        "description": (
                            "Music title. Limits: for V4 and V4_5ALL model - 80 chars, "
                            "for V4_5, V4_5PLUS and V5 - 100 chars"
                        ),
                    },
                    "negative_tags": {
                        "type": "string",
                        "description": (
                            "Music styles or traits to exclude from the generated audio."
                            "I.e.: 'Heavy Metal, Upbeat Drums'"
                        ),
                    },
                    "vocal_gender": {
                        "type": "string",
                        "enum": ["m", "f"],
                        "description": "Vocal gender for generated vocals. 'm' or 'f'",
                    },
                    "style_weight": {
                        "type": "number",
                        "description": "Weight of the provided style guidance. Range 0.00–1.00",
                        "default": 0.5,
                    },
                    "weirdness_constraint": {
                        "type": "number",
                        "description": "Constraint on creative deviation/novelty. Range 0.00–1.00",
                        "default": 0.5,
                    },
                    "instrumental": {
                        "type": "boolean",
                        "description": "Whether to generate instrumental only music.",
                        "default": False,
                    },
                    "suno_model": {
                        "type": "string",
                        "description": "Model version. Available options: V4, V4_5, V4_5PLUS, V4_5ALL, V5",
                        "default": "V5",
                    },
                    "execute_in_background": {
                        "type": "boolean",
                        "description": "Execute image generation in background.",
                        "default": True,
                    },
                },
                "required": ["prompt", "style", "title"],
            },
        ),
    )
    name = "generate_music_via_suno_custom_mode"

    @classmethod
    async def function(  # type: ignore[override]
        cls,
        prompt: str,
        style: str | None = None,
        title: str | None = None,
        suno_model: str = "V5",
        instrumental: bool = False,
        negative_tags: str | None = None,
        vocal_gender: str | None = None,
        style_weight: float = 0.5,
        weirdness_constraint: float = 0.5,
        execute_in_background: bool = True,
        **kwargs: Unpack[AdditionalOptions],
    ) -> dict[str, Any]:
        telegram_context = kwargs.get("telegram_context")
        telegram_update = kwargs.get("telegram_update")

        if telegram_context is None or telegram_update is None:
            raise ToolException(
                "This function requires telegram context & telegram update to be automatically provided."
            )
        if not style:
            raise ToolException("Style is mandatory in custom mode.")

        if not title:
            raise ToolException("Title is mandatory in custom mode.")

        if vocal_gender and vocal_gender not in ("f", "m"):
            raise ToolException("Vocal gender must be 'f' or 'm' if specified.")

        logger.log("TOOL", f"Generating music via Suno. Custom mode. Prompt: {prompt}. Style: {style}. Title: {title}")

        task_id = await cls._get_provider().order_music_generation_advanced_mode(
            prompt=prompt,
            instrumental_only=instrumental,
            model=suno_model,
            style=style,
            title=title,
            negative_tags=negative_tags,
            vocal_gender=vocal_gender,
            style_weight=style_weight,
            weirdness_constraint=weirdness_constraint,
        )

        logger.log("TOOL", f"[Task #{task_id}] Music generation request accepted by API.")

        coro = cls._poll_and_send_audio(
            task_id=task_id, telegram_context=telegram_context, telegram_update=telegram_update
        )

        if execute_in_background:
            logger.log("TOOL", f"[Task #{task_id}] Polling the generation result in the background...")
            task_manager.run_task(coro)
            return {
                "detail": "Music generation was successfully scheduled. User will receive it soon.",
                "suno_task_id": task_id,
            }

        return await coro
