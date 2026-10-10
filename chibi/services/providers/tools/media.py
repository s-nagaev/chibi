import mimetypes
from asyncio import sleep, to_thread
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Unpack

from loguru import logger
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition

from chibi.config import gpt_settings, telegram_settings
from chibi.schemas.app import ModelChangeSchema
from chibi.services.interface import UserInterface
from chibi.services.providers.constants.suno import POLLING_ATTEMPTS_WAIT_BETWEEN
from chibi.services.providers.tools.exceptions import ToolException
from chibi.services.providers.tools.tool import ChibiTool
from chibi.services.providers.tools.utils import AdditionalOptions, download
from chibi.services.user import generate_image, get_chibi_user, user_has_reached_images_generation_limit
from chibi.storage.files import get_file_storage

if TYPE_CHECKING:
    from chibi.services.providers import ElevenLabs, Minimax, Suno
    from chibi.services.providers.provider import Provider

MAX_REFERENCE_IMAGES = 10
MAX_REFERENCE_IMAGE_BYTES = 10 * 1024 * 1024
ALLOWED_REFERENCE_MIME_TYPES = frozenset({"image/png", "image/jpeg", "image/webp"})
UNKNOWN_MIME_TYPES = frozenset({None, "", "application/octet-stream"})
UNKNOWN_MIME_FALLBACK = "image/png"


class TextToSpeechTool(ChibiTool):
    register = True
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
        interface = cls.get_interface(kwargs=kwargs)
        logger.log("TOOL", "Sending voice message to user...")

        user = await get_chibi_user(user_id=interface.user_id)
        provider = user.tts_provider
        audio_data = await provider.speech(text=text)
        title = f"{text[:15]}..."

        await interface.send_audio(
            audio=audio_data,
            title=title,
            performer=f"{telegram_settings.bot_name} AI",
            filename=f"{title.replace(' ', '_')}.mp3",
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


def _normalize_reference_mime(mime: str | None) -> str:
    """Map unknown MIME types to the image/png fallback.

    Args:
        mime: MIME type reported by storage metadata or guessed from a file path.

    Returns:
        The original MIME type, or ``image/png`` when it is unknown.
    """
    if mime is None or mime in UNKNOWN_MIME_TYPES:
        return UNKNOWN_MIME_FALLBACK
    return mime


async def _resolve_reference_from_file_id(interface: UserInterface, file_id: str) -> tuple[bytes, str]:
    """Resolve a single reference image previously uploaded by the user.

    Args:
        interface: User interface bound to the current request.
        file_id: Unique identifier of a previously uploaded file.

    Returns:
        Tuple of raw image bytes and the resolved MIME type.

    Raises:
        ValueError: If the file storage backend is not configured for this interface.
        ToolException: If the file ID cannot be resolved from storage.
    """
    storage = get_file_storage(interface=interface)
    try:
        file_info = await storage.get_file_info(file_id=file_id)
        image_bytes = await storage.get_bytes(file_id=file_id)
    except FileNotFoundError as e:
        raise ToolException(f"Reference file with ID '{file_id}' was not found.") from e
    return image_bytes, _normalize_reference_mime(mime=file_info.get("mime_type"))


async def _resolve_reference_from_path(path_str: str) -> tuple[bytes, str]:
    """Resolve a single reference image from the local filesystem.

    Args:
        path_str: Absolute or tilde-expanded filesystem path to the image.

    Returns:
        Tuple of raw image bytes and the resolved MIME type.

    Raises:
        ToolException: If the path does not point to an existing file.
    """
    path = Path(path_str).expanduser().resolve()
    if not path.is_file():
        raise ToolException(f"Reference file not found: {path}")
    image_bytes = await to_thread(path.read_bytes)
    mime, _ = mimetypes.guess_type(str(path))
    return image_bytes, _normalize_reference_mime(mime=mime)


def _validate_reference_limits(images: list[tuple[bytes, str]]) -> None:
    """Validate resolved reference images against count, MIME and size constraints.

    Args:
        images: Resolved reference images as (bytes, mime) tuples.

    Raises:
        ToolException: If any limit is violated.
    """
    if len(images) > MAX_REFERENCE_IMAGES:
        raise ToolException(f"Too many reference images: {len(images)}. Maximum is {MAX_REFERENCE_IMAGES}.")
    for image_bytes, mime in images:
        if mime not in ALLOWED_REFERENCE_MIME_TYPES:
            raise ToolException(f"Unsupported reference image type '{mime}'. Allowed types: PNG, JPEG, WebP.")
        if len(image_bytes) > MAX_REFERENCE_IMAGE_BYTES:
            raise ToolException("Reference image exceeds the 10 MB size limit.")


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
                "and cannot be changed via the prompt. Reference images may be passed via reference_file_ids "
                "and/or reference_paths, but only models with image-to-image support accept them — the call "
                "fails for models without image input support. Allowed image formats: PNG, JPEG, WebP; "
                "maximum 10 reference images, 10 MB each."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "provider": {"type": "string", "description": "Provider name, i.e. 'Gemini'"},
                    "image_model": {"type": "string", "description": "Model name, i.e. 'dall-e-3'"},
                    "prompt": {"type": "string", "description": "Image generation prompt. English recommended"},
                    "reference_file_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "File IDs of previously uploaded images to use as reference input "
                            "(image-to-image). Optional. May be combined with reference_paths."
                        ),
                    },
                    "reference_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Local filesystem paths of images to use as reference input (image-to-image). "
                            "Optional. Requires filesystem access. May be combined with reference_file_ids."
                        ),
                    },
                },
                "required": ["provider", "image_model", "prompt"],
            },
        ),
    )
    name = "generate_image"

    @classmethod
    def _validate_image_input_support(cls, provider_name: str, model: str, provider: "Provider") -> None:
        """Ensure the selected provider and model accept reference images.

        Args:
            provider_name: Name of the provider selected for generation.
            model: Model name selected for generation.
            provider: Provider instance to inspect.

        Raises:
            ToolException: If the provider or the model does not support image input.
        """
        if not provider.image_to_image_ready:
            raise ToolException(f"Provider '{provider_name}' does not support image input for image generation.")
        if not provider.supports_image_input(model):
            raise ToolException(f"Model '{model}' of provider '{provider_name}' does not accept reference images.")

    @classmethod
    async def _resolve_reference_images(
        cls,
        interface: UserInterface,
        provider_name: str,
        model: str,
        reference_file_ids: list[str] | None,
        reference_paths: list[str] | None,
    ) -> list[tuple[bytes, str]] | None:
        """Validate and resolve optional reference images into bytes and MIME type tuples.

        Args:
            interface: User interface bound to the current request.
            provider_name: Name of the provider selected for generation.
            model: Model name selected for generation.
            reference_file_ids: File IDs of previously uploaded reference images, if any.
            reference_paths: Local filesystem paths of reference images, if any.

        Returns:
            Resolved reference images as (bytes, mime) tuples, or None when no reference
            input was provided.

        Raises:
            ToolException: If the selected provider or model does not accept image input,
                filesystem access is disabled, the total number of reference images exceeds
                the limit, a reference file is missing, or any image violates the MIME or
                size constraints.
        """
        file_ids = reference_file_ids or []
        paths = reference_paths or []
        if not file_ids and not paths:
            return None

        if len(file_ids) + len(paths) > MAX_REFERENCE_IMAGES:
            raise ToolException(
                f"Too many reference images: {len(file_ids) + len(paths)}. Maximum is {MAX_REFERENCE_IMAGES}."
            )

        user = await get_chibi_user(user_id=interface.user_id)
        provider = user.providers.get(provider_name=provider_name)
        if provider is None:
            raise ToolException(f"Provider '{provider_name}' is not available.")
        cls._validate_image_input_support(provider_name=provider_name, model=model, provider=provider)

        if paths and not gpt_settings.filesystem_access:
            raise ToolException(
                "Filesystem access is disabled. Local reference image paths require filesystem_access to be enabled."
            )

        images: list[tuple[bytes, str]] = []
        for file_id in file_ids:
            images.append(await _resolve_reference_from_file_id(interface=interface, file_id=file_id))
        for path_str in paths:
            images.append(await _resolve_reference_from_path(path_str=path_str))

        _validate_reference_limits(images=images)
        return images

    @classmethod
    async def generate_and_send_image(
        cls,
        provider: str,
        model: str,
        prompt: str,
        interface: UserInterface,
        images: list[tuple[bytes, str]] | None = None,
    ) -> None:
        """Generate an image and send it to the user.

        Args:
            provider: Provider name.
            model: Model name.
            prompt: Image generation prompt.
            interface: User interface bound to the current request.
            images: Optional reference images as (bytes, mime) tuples for models with
                image-to-image support.
        """
        generated_images = await generate_image(
            interface=interface, provider_name=provider, model=model, prompt=prompt, images=images
        )
        await interface.send_images(images=generated_images)
        return None

    @classmethod
    async def function(
        cls,
        provider: str,
        image_model: str,
        prompt: str,
        reference_file_ids: list[str] | None = None,
        reference_paths: list[str] | None = None,
        **kwargs: Unpack[AdditionalOptions],
    ) -> dict[str, str]:
        """Generate an image, optionally based on reference images, and send it to the user.

        Args:
            provider: Provider name, i.e. 'Gemini'.
            image_model: Model name, i.e. 'dall-e-3'.
            prompt: Image generation prompt.
            reference_file_ids: File IDs of previously uploaded images used as reference
                input. Supported only by image-to-image capable models. May be combined
                with reference_paths — a documented deviation from the vision tool's
                either-or precedent.
            reference_paths: Local filesystem paths of images used as reference input.
                Requires filesystem access to be enabled. May be combined with
                reference_file_ids.
            **kwargs: Additional options injected by the tool runner.

        Returns:
            Dict describing the operation outcome.

        Raises:
            ToolException: If the monthly limit is reached, the selected provider or model
                does not accept image input, filesystem access is disabled, the number of
                reference images exceeds the limit, a reference file is missing, or any
                image violates the MIME or size constraints.
        """
        interface = cls.get_interface(kwargs=kwargs)

        if await user_has_reached_images_generation_limit(user_id=interface.user_id):
            raise ToolException("User has reached image generation monthly limit.")

        images = await cls._resolve_reference_images(
            interface=interface,
            provider_name=provider,
            model=image_model,
            reference_file_ids=reference_file_ids,
            reference_paths=reference_paths,
        )

        await cls.generate_and_send_image(
            provider=provider,
            model=image_model,
            prompt=prompt,
            interface=interface,
            images=images,
        )
        return {"detail": "Image was successfully generated and sent to user."}


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
                        "description": "Model version. Available options: V4, V4_5, V4_5PLUS, V4_5ALL, V5, V5_5",
                        "default": "V5_5",
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
        suno_model: str = "V5_5",
        instrumental: bool = False,
        **kwargs: Unpack[AdditionalOptions],
    ) -> dict[str, Any]:
        interface = cls.get_interface(kwargs=kwargs)
        logger.log("TOOL", f"Generating music via Suno. Prompt: {prompt}")

        task_id = await cls._get_provider().order_music_generation(
            prompt=prompt, instrumental_only=instrumental, model=suno_model
        )

        logger.log("TOOL", f"[Task #{task_id}] Music generation request accepted by API.")

        return await cls._poll_and_send_audio(task_id=task_id, interface=interface)

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
        interface: UserInterface,
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

            logger.log("TOOL", f"[Suno] Audio and thumbnail downloaded. Sending it to the chat #{interface.chat_id}...")
            await interface.send_audio(
                audio=await download(str(music_url)) or str(music_url),
                title=title,
                performer=f"{telegram_settings.bot_name} AI via Suno AI",
                duration=int(suno_data.duration) if suno_data.duration else None,
                thumbnail=await download(str(image_url)) if image_url else None,
                filename=f"{title.replace(' ', '_')}.mp3",
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
                            "V4_5ALL, V5 and V5_5 - 5000 chars."
                        ),
                    },
                    "style": {
                        "type": "string",
                        "description": (
                            "Music style, i.e. 'Uplifting trance, Techno, Eurodance'"
                            "Limits: for V4 model - 200 chars, for V4_5, V4_5PLUS, "
                            "V4_5ALL, V5 and V5_5 - 1000 chars"
                        ),
                    },
                    "title": {
                        "type": "string",
                        "description": (
                            "Music title. Limits: for V4 and V4_5ALL model - 80 chars, "
                            "for V4_5, V4_5PLUS, V5 and V5_5 - 100 chars"
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
                        "description": "Model version. Available options: V4, V4_5, V4_5PLUS, V4_5ALL, V5, V5_5",
                        "default": "V5_5",
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
        suno_model: str = "V5_5",
        instrumental: bool = False,
        negative_tags: str | None = None,
        vocal_gender: str | None = None,
        style_weight: float = 0.5,
        weirdness_constraint: float = 0.5,
        **kwargs: Unpack[AdditionalOptions],
    ) -> dict[str, Any]:
        interface = cls.get_interface(kwargs=kwargs)
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

        result = await cls._poll_and_send_audio(task_id=task_id, interface=interface)
        return result


class GenerateMusicViaElevenLabsTool(ChibiTool):
    register = bool(gpt_settings.elevenlabs_api_key)
    run_in_background_by_default: bool = True
    allow_model_to_change_background_mode: bool = False
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="generate_music_via_elevenlabs",
            description=(
                "Generate music via ElevenLabs Music API and send it to the user as an audio file. "
                "You won't hear the audio itself, only a message about whether the operation was successful or not."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Description of the music to generate. English recommended.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Music title. English recommended.",
                    },
                    "music_length_ms": {
                        "type": "integer",
                        "description": (
                            "Length of the generated track in milliseconds. "
                            "Supported range: 10000 (10 s) - 600000 (10 min). Default: 180000 (3 min)."
                        ),
                        "default": 180000,
                    },
                },
                "required": ["prompt", "title"],
            },
        ),
    )
    name = "generate_music_via_elevenlabs"
    _provider: Optional["ElevenLabs"] = None

    @classmethod
    async def function(
        cls,
        prompt: str,
        music_length_ms: int = 180000,
        title: str | None = None,
        **kwargs: Unpack[AdditionalOptions],
    ) -> dict[str, Any]:
        interface = cls.get_interface(kwargs=kwargs)
        music_length_ms = max(10000, min(music_length_ms, 600000))
        logger.log("TOOL", f"Generating music via ElevenLabs ({music_length_ms} ms). Prompt: {prompt}")

        audio = await cls._get_provider().generate_music(prompt=prompt, music_length_ms=music_length_ms)

        name = title or f"{prompt[:18]}..."
        logger.log("TOOL", f"[ElevenLabs] Music generated. Sending it to the chat #{interface.chat_id}...")
        await interface.send_audio(
            audio=audio,
            title=name,
            performer=f"{telegram_settings.bot_name} AI via ElevenLabs",
            duration=music_length_ms // 1000,
            filename=f"{name.replace(' ', '_')}.mp3",
        )
        return {"detail": "Music was successfully generated and sent to user"}

    @classmethod
    def _get_provider(cls) -> "ElevenLabs":
        from chibi.services.providers import ElevenLabs, RegisteredProviders

        if cls._provider:
            return cls._provider
        provider = RegisteredProviders().get(provider_name="ElevenLabs")
        if not isinstance(provider, ElevenLabs):
            raise ToolException("This function requires ElevenLabs provider to be set.")
        cls._provider = provider
        return provider


class GenerateMusicViaMinimaxTool(ChibiTool):
    register = False  # MiniMax music generation endpoint deactivated by vendor; re-enable when available
    run_in_background_by_default: bool = True
    allow_model_to_change_background_mode: bool = False
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="generate_music_via_minimax",
            description=(
                "Generate music via MiniMax Music API and send it to the user as an audio file. "
                "You won't hear the audio itself, only a message about whether the operation was successful or not."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Description of the music to generate. English recommended.",
                    },
                },
                "required": ["prompt"],
            },
        ),
    )
    name = "generate_music_via_minimax"
    _provider: Optional["Minimax"] = None

    @classmethod
    async def function(
        cls,
        prompt: str,
        **kwargs: Unpack[AdditionalOptions],
    ) -> dict[str, Any]:
        interface = cls.get_interface(kwargs=kwargs)
        logger.log("TOOL", f"Generating music via MiniMax. Prompt: {prompt}")

        audio = await cls._get_provider().generate_music(prompt=prompt)

        title = f"{prompt[:15]}..."
        logger.log("TOOL", f"[MiniMax] Music generated. Sending it to the chat #{interface.chat_id}...")
        await interface.send_audio(
            audio=audio,
            title=title,
            performer=f"{telegram_settings.bot_name} AI via MiniMax",
            filename=f"{title.replace(' ', '_')}.mp3",
        )
        return {"detail": "Music was successfully generated and sent to user"}

    @classmethod
    def _get_provider(cls) -> "Minimax":
        from chibi.services.providers import Minimax, RegisteredProviders

        if cls._provider:
            return cls._provider
        provider = RegisteredProviders().get(provider_name="Minimax")
        if not isinstance(provider, Minimax):
            raise ToolException("This function requires Minimax provider to be set.")
        cls._provider = provider
        return provider
