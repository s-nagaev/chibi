from asyncio import sleep
from typing import TYPE_CHECKING, Any, Optional, Unpack

from loguru import logger
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition
from telegram import Update
from telegram.ext import ContextTypes

from chibi.config import gpt_settings, telegram_settings
from chibi.constants import AUDIO_UPLOAD_TIMEOUT
from chibi.schemas.app import ModelChangeSchema
from chibi.services.providers.constants.suno import POLLING_ATTEMPTS_WAIT_BETWEEN
from chibi.services.providers.tools.exceptions import ToolException
from chibi.services.providers.tools.tool import ChibiTool
from chibi.services.providers.tools.utils import AdditionalOptions, download
from chibi.services.user import generate_image, get_chibi_user, user_has_reached_images_generation_limit
from chibi.utils.telegram import get_telegram_chat

if TYPE_CHECKING:
    from chibi.services.providers import Suno


class TextToSpeechTool(ChibiTool):
    register = bool(gpt_settings.openai_key)
    run_in_background_by_default = True
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
        user_id = kwargs.get("user_id")
        if not user_id:
            raise ToolException("This function requires user_id to be automatically provided.")

        telegram_context = kwargs.get("telegram_context")
        telegram_update = kwargs.get("telegram_update")

        if telegram_context is None or telegram_update is None:
            raise ToolException(
                "This function requires telegram context & telegram update to be automatically provided."
            )
        logger.log("TOOL", "Sending voice message to user...")

        user = await get_chibi_user(user_id=user_id)
        provider = user.tts_provider
        audio_data = await provider.speech(text=text)
        title = f"{text[:15]}..."
        await telegram_context.bot.send_audio(
            chat_id=get_telegram_chat(update=telegram_update).id,
            audio=audio_data,
            title=title,
            performer=f"{telegram_settings.bot_name} AI",
            filename=f"{title.replace(' ', '_')}.mp3",
            parse_mode="HTML",
            read_timeout=AUDIO_UPLOAD_TIMEOUT,
            write_timeout=AUDIO_UPLOAD_TIMEOUT,
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


class GenerateImageTool(ChibiTool):
    register = True
    run_in_background_by_default = True
    allow_model_to_change_background_mode = False
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

        await cls.generate_and_send_image(
            user_id=user_id,
            provider=provider,
            model=image_model,
            prompt=prompt,
            update=telegram_update,
            context=telegram_context,
        )
        return {"detail": "Image was successfully generated and sent."}


class GenerateMusicViaSunoTool(ChibiTool):
    register = bool(gpt_settings.suno_key)
    run_in_background_by_default: bool = True
    allow_model_to_change_background_mode: bool = False
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

        return await cls._poll_and_send_audio(
            task_id=task_id, telegram_context=telegram_context, telegram_update=telegram_update
        )

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
        logger.log("TOOL", f"[Task #{task_id}] Polling the generation result...")
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
        generated_data: dict[str | int, Any] = {"task_id": task_id}

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

            image_url = suno_data.source_image_url or suno_data.image_url
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
            generated_version_data = {
                "id": suno_data.id,
                "title": title,
                "music_url": str(music_url),
                "image_url": str(image_url),
            }
            generated_data[version] = generated_version_data
        return {
            "detail": "Music was successfully generated and sent to user",
            "suno_task_data": generated_data,
        }


class GenerateAdvancedMusicViaSunoTool(GenerateMusicViaSunoTool):
    run_in_background_by_default: bool = True
    allow_model_to_change_background_mode: bool = False
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

        result = await cls._poll_and_send_audio(
            task_id=task_id, telegram_context=telegram_context, telegram_update=telegram_update
        )
        return result
