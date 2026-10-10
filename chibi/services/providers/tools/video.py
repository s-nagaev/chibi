import re
from io import BytesIO
from typing import Any, Unpack

from loguru import logger
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition

from chibi.config import gpt_settings
from chibi.schemas.app import ModelChangeSchema
from chibi.services.interface import UserInterface
from chibi.services.providers.tools.exceptions import ToolException
from chibi.services.providers.tools.tool import ChibiTool
from chibi.services.providers.tools.utils import AdditionalOptions
from chibi.services.user import (
    generate_video,
    get_models_available,
    user_has_reached_videos_generation_limit,
)

# Telegram delivers videos as video messages up to ~48 MB; larger payloads are
# sent as documents instead.
VIDEO_SIZE_GUARD_BYTES = 48 * 1024 * 1024
# Telegram video thumbnails must be JPEG, smaller than 200 kB and at most
# 320 px on the longer side.
THUMBNAIL_MAX_BYTES = 200 * 1024
THUMBNAIL_MAX_PIXELS = 320


VIDEO_TOOL_NAMES = ["generate_video", "get_available_video_generation_models"]


def sync_video_tool_registration() -> None:
    """Align the video tools' registration with the operator configuration.

    Mirrors the image tools: the tool classes are statically registered (the
    ``ChibiTool.__init_subclass__`` whitelist gate applied at class-definition
    time), and in public mode availability is resolved per user at call time.
    This sync pass only matters in private mode, where the tools make sense
    solely when at least one configured provider is video-ready. The
    ``TOOLS_WHITELIST`` gate is honoured exactly like
    ``ChibiTool.__init_subclass__`` does.

    Called at the end of the ``chibi.services.providers`` package import, once
    the provider registry is fully populated.
    """
    from chibi.services.providers import RegisteredProviders
    from chibi.services.providers.tools.tool import RegisteredChibiTools

    if not gpt_settings.public_mode and not RegisteredProviders().video_generation_ready:
        RegisteredChibiTools.deregister_tools(VIDEO_TOOL_NAMES)
        return None

    for tool in (GetAvailableVideoGenerationModelsTool, GenerateVideoTool):
        if gpt_settings.tools_whitelist and tool.name not in gpt_settings.tools_whitelist:
            RegisteredChibiTools.deregister_tools([tool.name])
            continue
        if tool.name not in RegisteredChibiTools.tools_map:
            RegisteredChibiTools.register(tool)
    return None


def prepare_video_thumbnail(thumbnail: bytes) -> bytes | None:
    """Fit a provider thumbnail to Telegram's video-thumbnail constraints.

    Telegram expects a JPEG below 200 kB and at most 320 px on each side.
    Oversized or wrongly encoded thumbnails are downscaled and re-encoded;
    if that is not possible, ``None`` is returned so the video is sent
    without a thumbnail.

    Args:
        thumbnail: Raw thumbnail bytes as returned by the provider.

    Returns:
        Thumbnail bytes meeting the constraints, or ``None``.
    """
    try:
        from PIL import Image
        from PIL.ImageFile import ImageFile

        image: ImageFile
        with Image.open(BytesIO(thumbnail)) as loaded:
            loaded.load()
            image = loaded
            if image.width > THUMBNAIL_MAX_PIXELS or image.height > THUMBNAIL_MAX_PIXELS:
                image.thumbnail((THUMBNAIL_MAX_PIXELS, THUMBNAIL_MAX_PIXELS))
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")  # type: ignore[assignment]
            for quality in (85, 70, 55, 40):
                buffer = BytesIO()
                image.save(buffer, format="JPEG", quality=quality)
                if buffer.tell() < THUMBNAIL_MAX_BYTES:
                    return buffer.getvalue()
    except Exception as exc:
        logger.warning(f"Video thumbnail could not be prepared and is dropped: {exc}")
    return None


class GetAvailableVideoGenerationModelsTool(ChibiTool):
    register = True
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="get_available_video_generation_models",
            description=("Get models and providers available for user for video generation."),
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    )
    name = "get_available_video_generation_models"

    @classmethod
    async def function(cls, **kwargs: Unpack[AdditionalOptions]) -> dict[str, Any]:
        user_id = kwargs.get("user_id")
        if not user_id:
            raise ToolException("This function requires user_id to be automatically provided.")

        logger.log("TOOL", f"Getting available video generation models for user {user_id}...")

        data: list[ModelChangeSchema] = await get_models_available(user_id=user_id, video_generation=True)

        return {
            "available_models": [info.model_dump(include={"provider", "name", "display_name"}) for info in data],
        }


class GenerateVideoTool(ChibiTool):
    register = True
    run_in_background_by_default = True
    allow_model_to_change_background_mode = False
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="generate_video",
            description=(
                "Generate video using one of the available models. You won’t see the video itself, only a message "
                "about whether the operation was successful or not. Check available providers and models first. "
                "Use your knowledge to adapt the prompt for a specific model to achieve the best result. "
                "The aspect ratio and resolution are determined by the chosen model and cannot be changed via "
                "the prompt; the duration can be requested via the duration parameter if the model supports it."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "description": (
                            "Provider name, i.e. 'Alibaba'. Optional; defaults to the user's active video provider."
                        ),
                    },
                    "video_model": {
                        "type": "string",
                        "description": (
                            "Model name, i.e. 'wan2.6-t2v'. Optional; defaults to the provider's video model."
                        ),
                    },
                    "prompt": {"type": "string", "description": "Video generation prompt. English recommended"},
                    "duration": {
                        "type": "integer",
                        "description": "Length of the video in seconds, if the model supports choosing it.",
                    },
                },
                "required": ["prompt"],
            },
        ),
    )
    name = "generate_video"

    @classmethod
    async def generate_and_send_video(
        cls,
        provider: str | None,
        model: str | None,
        prompt: str,
        duration: int | None,
        interface: UserInterface,
    ) -> None:
        result = await generate_video(
            interface=interface,
            prompt=prompt,
            model=model,
            provider_name=provider,
            duration=duration,
        )

        thumbnail = prepare_video_thumbnail(result.thumbnail) if result.thumbnail else None
        title = f"{prompt[:20]}..."
        # Keep the prompt-derived filename filesystem-safe: strip slashes,
        # newlines, quotes, emoji and other non-word characters.
        safe_name = re.sub(r"[^\w-]+", "_", prompt[:40]).strip("_") or "video"
        filename = f"{safe_name}.mp4"

        if len(result.video) > VIDEO_SIZE_GUARD_BYTES:
            logger.log(
                "TOOL",
                f"[Video] Payload of {len(result.video)} bytes exceeds the video size guard. "
                f"Sending it to the chat #{interface.chat_id} as document...",
            )
            await interface.send_document(
                document=result.video,
                filename=filename,
                caption="The video is delivered as a file because of its size.",
                thumbnail=thumbnail,
            )
            return None

        logger.log("TOOL", f"[Video] Video generated. Sending it to the chat #{interface.chat_id}...")
        await interface.send_video(
            video=result.video,
            title=title,
            duration=result.duration,
            thumbnail=thumbnail,
            filename=filename,
        )
        return None

    @classmethod
    async def function(
        cls,
        prompt: str,
        provider: str | None = None,
        video_model: str | None = None,
        duration: int | None = None,
        **kwargs: Unpack[AdditionalOptions],
    ) -> dict[str, str]:
        interface = cls.get_interface(kwargs=kwargs)

        if await user_has_reached_videos_generation_limit(user_id=interface.user_id):
            raise ToolException(
                "User has reached video generation monthly limit. "
                "Please check the available models and try again next month."
            )

        await cls.generate_and_send_video(
            provider=provider,
            model=video_model,
            prompt=prompt,
            duration=duration,
            interface=interface,
        )
        return {"detail": "Video was successfully generated and sent to user."}
