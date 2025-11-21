from typing import Any, Unpack

from loguru import logger
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition

from chibi.config import gpt_settings
from chibi.schemas.app import ModelChangeSchema
from chibi.services.providers.tools.exceptions import ToolException
from chibi.services.providers.tools.tool import ChibiTool
from chibi.services.providers.tools.utils import AdditionalOptions
from chibi.services.user import generate_image, user_has_reached_images_generation_limit


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
        from chibi.utils import send_audio

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
            "available_models": [info.model_dump(include={"provider", "name"}) for info in data],
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
                    "model": {"type": "string", "description": "Model name, i.e. 'dall-e-3'"},
                    "prompt": {"type": "string", "description": "Image generation prompt. English recommended"},
                },
                "required": ["provider", "model", "prompt"],
            },
        ),
    )
    name = "generate_image"

    @classmethod
    async def function(
        cls, provider: str, model: str, prompt: str, **kwargs: Unpack[AdditionalOptions]
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

        from chibi.utils import send_images

        if await user_has_reached_images_generation_limit(user_id=user_id):
            raise ToolException("User has reached image generation monthly limit.")

        images = await generate_image(user_id=user_id, provider_name=provider, model=model, prompt=prompt)
        await send_images(images=images, update=telegram_update, context=telegram_context)

        return {"detail": "Image was successfully generated and sent."}
