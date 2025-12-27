from typing import Any, Unpack

from loguru import logger
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition
from telegram import Update
from telegram.ext import ContextTypes

from chibi.config import gpt_settings
from chibi.schemas.app import ChatResponseSchema, ModelChangeSchema
from chibi.services.providers.tools.exceptions import ToolException
from chibi.services.providers.tools.tool import ChibiTool
from chibi.services.providers.tools.utils import AdditionalOptions, get_sub_agent_response
from chibi.services.task_manager import task_manager
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
        from chibi.utils import send_images

        images = await generate_image(user_id=user_id, provider_name=provider, model=model, prompt=prompt)
        await send_images(images=images, update=update, context=context)
        return None

    @classmethod
    async def function(
        cls,
        provider: str,
        image_model: str,
        prompt: str,
        execute_in_background: bool = False,
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
                "information, to avoid saturating your context."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Prompt"},
                },
                "required": ["prompt"],
            },
        ),
    )
    name = "delegate_task"

    @classmethod
    async def function(cls, prompt: str, **kwargs: Unpack[AdditionalOptions]) -> dict[str, str]:
        user_id = kwargs.get("user_id")
        if not user_id:
            raise ToolException("This function requires user_id to be automatically provided.")
        model = kwargs.get("model")
        if not model:
            raise ToolException("This function requires model to be automatically provided.")
        logger.log("DELEGATE", f"Model {model} delegating a task: {prompt}")

        response: ChatResponseSchema = await get_sub_agent_response(user_id=user_id, prompt=prompt)
        logger.log("SUBAGENT", f"Delegated task is done: {response.answer}")

        return {"response": response.answer}


class SendTextFileTool(ChibiTool):
    register = True
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="send_text_based_file",
            description=("Send a data as a text-based file (.md, .txt, .rst, .py, etc)  to user."),
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

        from chibi.utils import send_text_file

        await send_text_file(file_content=content, file_name=filename, update=telegram_update, context=telegram_context)

        return {"detail": "File was successfully sent."}
