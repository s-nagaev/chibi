import base64
import random
from asyncio import sleep
from copy import copy
from io import BytesIO
from uuid import uuid4

from google.genai.client import Client
from google.genai.types import (
    ContentDict,
    FunctionCallDict,
    FunctionDeclaration,
    FunctionResponseDict,
    GenerateContentConfig,
    GenerateContentResponse,
    PartDict,
    Tool,
)
from loguru import logger
from telegram import Update
from telegram.ext import ContextTypes

from chibi.config import gpt_settings
from chibi.exceptions import ServiceResponseError
from chibi.models import Message, User
from chibi.schemas.app import ChatResponseSchema, ModelChangeSchema
from chibi.services.providers.provider import RestApiFriendlyProvider
from chibi.services.providers.tools import registered_functions, tools
from chibi.services.providers.utils import (
    escape_and_truncate,
    get_usage_from_google_response,
    get_usage_msg,
    prepare_system_prompt,
    send_llm_thoughts,
)


class Gemini(RestApiFriendlyProvider):
    api_key = gpt_settings.gemini_key
    chat_ready = True
    image_generation_ready = True

    name = "Gemini"
    model_name_keywords = ["gemini", "gemma"]
    model_name_keywords_exclude = ["image", "vision", "tts", "embedding", "2.0", "1.5"]
    default_model = "models/gemini-2.5-flash"
    default_image_model = "models/imagen-4.0-fast-generate-001"
    frequency_penalty: float | None = gpt_settings.frequency_penalty
    max_tokens: int = gpt_settings.max_tokens
    presence_penalty: float | None = gpt_settings.presence_penalty
    temperature: float = gpt_settings.temperature

    def __init__(self, token: str) -> None:
        super().__init__(token=token)

    @property
    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json"}

    @property
    def tools_list(self) -> list[Tool]:
        """Convert our tools format to Google's Tool format.

        Returns:
            Tools list in Google's Tool format.
        """
        google_tools = []
        for tool in tools:
            google_tool = Tool(
                function_declarations=[
                    FunctionDeclaration(
                        name=tool["function"]["name"],
                        description=tool["function"]["description"],
                        parameters=tool["function"]["parameters"],
                    )
                ]
            )
            google_tools.append(google_tool)
        return google_tools

    def _get_text(self, response: GenerateContentResponse) -> str | None:
        if not response.candidates or not response.candidates[0].content or not response.candidates[0].content.parts:
            return None
        text = ""
        for part in response.candidates[0].content.parts:
            if part.text is not None and not part.thought:
                text += part.text
        return text if text != "" else None

    async def _generate_content(
        self, model: str, contents: list[ContentDict], config: GenerateContentConfig
    ) -> GenerateContentResponse:
        for attempt in range(gpt_settings.retries):
            async with Client(api_key=gpt_settings.gemini_key).aio as client:
                response: GenerateContentResponse = await client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
            answer = self._get_text(response)
            if answer is not None or response.function_calls:
                return response

            delay = gpt_settings.backoff_factor * (2**attempt)
            jitter = delay * random.uniform(0.1, 0.5)
            total_delay = delay + jitter

            logger.warning(
                f"Attempt #{attempt + 1}. Unexpected (empty) response received. Retrying in {total_delay} seconds..."
            )
            await sleep(total_delay)
        raise ServiceResponseError(provider=self.name, model=model, detail="Unexpected (empty) response received")

    async def _get_chat_completion_response(
        self,
        messages: list[ContentDict],
        user: User | None = None,
        model: str | None = None,
        system_prompt: str = gpt_settings.assistant_prompt,
        context: ContextTypes.DEFAULT_TYPE | None = None,
        update: Update | None = None,
    ) -> tuple[ChatResponseSchema, list[ContentDict]]:
        model_name = model or self.default_model

        prepared_system_prompt = (
            await prepare_system_prompt(base_system_prompt=system_prompt, user=user) if user else system_prompt
        )

        generation_config = GenerateContentConfig(
            system_instruction=prepared_system_prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            tools=self.tools_list if "gemini" in model_name else None,
        )

        response = await self._generate_content(
            model=model_name,
            contents=messages,
            config=generation_config,
        )
        answer = self._get_text(response)

        usage = get_usage_from_google_response(response_message=response)
        usage_message = get_usage_msg(usage=usage)

        if not response.function_calls:
            messages.append(
                ContentDict(
                    role="model",
                    parts=[
                        PartDict(
                            text=answer,
                        )
                    ],
                )
            )
            return ChatResponseSchema(answer=answer, provider=self.name, model=model_name, usage=usage), messages

        thoughts = answer or "No thoughts..."
        logger.log("THINK", f"{model_name}: {thoughts[:1800]}. {usage_message}")

        if all((context is not None, update is not None, bool(answer))):
            await send_llm_thoughts(thoughts=thoughts, context=context, update=update)

        for function_call in response.function_calls:
            function_to_call = registered_functions.get(str(function_call.name))
            if not function_to_call:
                logger.error(f"Function {str(function_call.name)} called but it's not registered.")
                function_to_call = registered_functions["stub_function"]

            function_args = copy(function_call.args) if function_call.args else {}
            function_args["user_id"] = user.id if user else 0
            function_args["telegram_context"] = context
            function_args["telegram_update"] = update

            function_call_id = function_call.id or str(uuid4())

            logger.log(
                "CALL",
                f"Calling a function '{function_call.name}'. Args: {escape_and_truncate(function_call.args)}",
            )
            function_response = await function_to_call(**function_args)

            try:
                thought_signature = response.candidates[0].content.parts[0].thought_signature  # type: ignore
            except Exception as e:
                logger.error(
                    f"Could not get thought signature for function {function_call.name} call: {e}. "
                    f"{response.candidates[0] if response.candidates else 'No response candidates found'}."
                )
                thought_signature = None

            tool_call_message: ContentDict = ContentDict(
                role="model",
                parts=[
                    PartDict(
                        # text=response.text,
                        function_call=FunctionCallDict(
                            name=function_call.name, args=function_call.args, id=function_call_id
                        ),
                        thought_signature=thought_signature,
                    ),
                ],
            )

            tool_result_message = ContentDict(
                role="user",
                parts=[
                    PartDict(
                        function_response=FunctionResponseDict(
                            id=function_call_id, name=function_call.name, response=function_response.model_dump()
                        )
                    ),
                ],
            )

            messages.append(tool_call_message)
            messages.append(tool_result_message)
            logger.log("CALL", f"Function result received, returning it to the model {model_name}...")
            return await self._get_chat_completion_response(
                messages=messages,
                model=model_name,
                user=user,
                system_prompt=system_prompt,
                context=context,
                update=update,
            )
        # TODO: unreachable raise, temporary solution, for mypy only
        raise ServiceResponseError(provider=self.name, model=model_name, detail="this should never happen")

    async def get_chat_response(
        self,
        messages: list[Message],
        user: User | None = None,
        model: str | None = None,
        system_prompt: str = gpt_settings.assistant_prompt,
        update: Update | None = None,
        context: ContextTypes.DEFAULT_TYPE | None = None,
    ) -> tuple[ChatResponseSchema, list[Message]]:
        model = model or self.default_model
        initial_messages = [msg.to_google() for msg in messages]

        chat_response, updated_messages = await self._get_chat_completion_response(
            messages=initial_messages,
            user=user,
            model=model,
            system_prompt=system_prompt,
            context=context,
            update=update,
        )

        new_messages = [msg for msg in updated_messages if msg not in initial_messages]
        return chat_response, [Message.from_google(msg) for msg in new_messages]

    async def get_images(self, prompt: str, model: str | None = None) -> list[BytesIO]:
        base_url = "https://generativelanguage.googleapis.com/v1beta/"
        model = model or self.default_image_model
        params = {"key": self.token}

        if "image-gen" in model:
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseModalities": ["Text", "Image"]},
            }
            url = f"{base_url}{model}:generateContent"

        else:
            payload = {
                "instances": [{"prompt": prompt}],
                "parameters": {
                    "sampleCount": gpt_settings.image_n_choices if "ultra" not in model else 1,
                    "aspectRatio": gpt_settings.image_aspect_ratio,
                },
            }
            url = f"{base_url}{model}:predict"

        response = await self._request(method="POST", url=url, data=payload, params=params)
        response_data = response.json()
        if "image-gen" in model:
            image_data_b64 = response_data["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
            return [BytesIO(base64.b64decode(image_data_b64))]

        images = [x["bytesBase64Encoded"] for x in response_data["predictions"]]
        return [BytesIO(base64.b64decode(image_data_b64)) for image_data_b64 in images]

    @classmethod
    def is_image_ready_model(cls, model_name: str) -> bool:
        return "image" in model_name

    def get_model_display_name(self, model_name: str) -> str:
        if "gemini-3-pro-image" in model_name:
            return "Nano Banana Pro"
        if "imagen" in model_name:
            model_name = model_name.replace("generate-", "")
        return model_name[7:].replace("-", " ")

    async def get_available_models(self, image_generation: bool = False) -> list[ModelChangeSchema]:
        try:
            async with Client(api_key=gpt_settings.gemini_key).aio as aclient:
                models = await aclient.models.list()
        except Exception as e:
            logger.error(f"Failed to get available models for provider {self.name} due to exception: {e}")
            return []

        all_models = [
            ModelChangeSchema(
                provider=self.name,
                name=model.name,
                display_name=self.get_model_display_name(model.name),
                image_generation=self.is_image_ready_model(model.name),
            )
            async for model in models
            if model.name
        ]
        all_models.sort(key=lambda model: model.name)

        if image_generation:
            return [model for model in all_models if model.image_generation]

        if gpt_settings.models_whitelist:
            return [model for model in all_models if model.name in gpt_settings.models_whitelist]

        return [model for model in all_models if self.is_chat_ready_model(model.name)]
