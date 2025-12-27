import asyncio
import math
import random
from asyncio import sleep
from copy import copy
from io import BytesIO
from typing import Any
from uuid import uuid4

from google.genai.client import Client
from google.genai.errors import APIError
from google.genai.types import (
    ContentDict,
    FunctionCallDict,
    FunctionDeclaration,
    FunctionResponseDict,
    GenerateContentConfig,
    GenerateContentResponse,
    GenerateImagesConfig,
    GenerateImagesResponse,
    HttpOptions,
    Image,
    ImageConfig,
    PartDict,
    Tool,
)
from loguru import logger
from telegram import Update
from telegram.ext import ContextTypes

from chibi.config import application_settings, gpt_settings
from chibi.exceptions import NoResponseError, NotAuthorizedError, ServiceRateLimitError, ServiceResponseError
from chibi.models import Message, User
from chibi.schemas.app import ChatResponseSchema, ModelChangeSchema
from chibi.services.metrics import MetricsService
from chibi.services.providers.provider import RestApiFriendlyProvider
from chibi.services.providers.tools import RegisteredChibiTools
from chibi.services.providers.utils import (
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
    def tools_list(self) -> list[Tool]:
        """Convert our tools format to Google's Tool format.

        Returns:
            Tools list in Google's Tool format.
        """
        google_tools = []
        for tool in RegisteredChibiTools.get_tool_definitions():
            try:
                google_tool = Tool(
                    function_declarations=[
                        FunctionDeclaration(
                            name=str(tool["function"]["name"]),
                            description=str(tool["function"]["description"]),
                            parameters=tool["function"]["parameters"],
                        )
                    ]
                )
            except Exception as e:
                logger.error(f"Failed to register tool {tool['function']['name']} due to exception: {e}")
                import pprint

                pprint.pprint(tool)
                raise
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

    def _get_thought_signature(self, response: GenerateContentResponse) -> bytes | None:
        if not response.candidates:
            return None
        first_candidate = response.candidates[0]
        if not first_candidate.content or not first_candidate.content.parts:
            return None
        for part in first_candidate.content.parts:
            if signature := part.thought_signature:
                return signature
        return None

    def _get_retry_delay(self, response: Any) -> float | None:
        if not isinstance(response, dict):
            logger.warning(
                f"The Gemini API error response data is not a dict. Skipping getting retry delay. "
                f"Response type: {type(response)}. Response: {response}"
            )
            return None
        per_day_quota: bool = False
        retry_delay = None
        details = response.get("error", {}).get("details", [])
        if not details:
            logger.warning(
                f"The Gemini API error response data does not contain details section. Skipping getting retry delay. "
                f"Response: {response}"
            )
            return None

        for item in details:
            detail_type = item.get("@type", "")

            if "QuotaFailure" in detail_type:
                violations = item.get("violations", [])
                for v in violations:
                    if "PerDay" in v.get("quotaId", ""):
                        per_day_quota = True
                        break

            elif "RetryInfo" in detail_type:
                delay_str = item.get("retryDelay", "")
                if delay_str and delay_str.endswith("s"):
                    try:
                        val = float(delay_str[:-1])
                        retry_delay = math.ceil(val) + 1
                    except ValueError:
                        retry_delay = None
        if per_day_quota:
            logger.warning("Ooops! Seems we have reached Daily Quota for Gemini API.")
            return None
        return retry_delay

    async def _generate_content(
        self, model: str, contents: list[ContentDict], config: GenerateContentConfig
    ) -> GenerateContentResponse:
        for attempt in range(gpt_settings.retries):
            try:
                async with Client(api_key=gpt_settings.gemini_key).aio as client:
                    response: GenerateContentResponse = await client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config,
                    )
                answer = self._get_text(response)
                if answer is not None or response.function_calls:
                    return response
            except APIError as err:
                logger.error(f"Gemini API error: {err.message}")

                if err.code == 429:
                    retry_delay = self._get_retry_delay(err.details)
                    if not retry_delay:
                        raise ServiceRateLimitError(provider=self.name, model=model, detail=err.details)
                    await asyncio.sleep(retry_delay + random.uniform(0.5, 2.5))
                    continue

                elif err.code == 403:
                    raise NotAuthorizedError(provider=self.name, model=model, detail=err.details)

                else:
                    raise ServiceResponseError(provider=self.name, model=model, detail=err.details)

            delay = gpt_settings.backoff_factor * (2**attempt)
            jitter = delay * random.uniform(0.1, 0.5)
            total_delay = delay + jitter

            logger.warning(
                f"Attempt #{attempt + 1}. Unexpected (empty) response received. Retrying in {total_delay} seconds..."
            )
            await sleep(total_delay)
        raise NoResponseError(provider=self.name, model=model, detail="Unexpected (empty) response received")

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

        prepared_system_prompt = await prepare_system_prompt(base_system_prompt=system_prompt, user=user)

        if "flash" in model_name and self.temperature > 0.4:
            temperature = 0.4
        else:
            temperature = self.temperature

        http_options = HttpOptions(httpx_async_client=self.get_async_httpx_client())

        generation_config = GenerateContentConfig(
            system_instruction=prepared_system_prompt if "gemini" in model_name else None,
            temperature=temperature,
            max_output_tokens=self.max_tokens,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            tools=self.tools_list if "gemini" in model_name else None,
            http_options=http_options,
        )

        response: GenerateContentResponse = await self._generate_content(
            model=model_name,
            contents=messages,
            config=generation_config,
        )
        answer = self._get_text(response)
        usage = get_usage_from_google_response(response_message=response)
        if application_settings.is_influx_configured:
            MetricsService.send_usage_metrics(metric=usage, model=model_name, provider=self.name, user=user)
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

        # Tool calls handling
        logger.log("CALL", f"LLM requested the call of {len(response.function_calls)} tools.")

        if answer:
            await send_llm_thoughts(thoughts=answer, context=context, update=update)
        logger.log("THINK", f"{model}: {answer or 'No thoughts...'}. {usage_message}")

        tool_context: dict[str, Any] = {
            "user_id": user.id if user else None,
            "telegram_context": context,
            "telegram_update": update,
            "model": model,
        }

        tool_coroutines = [
            RegisteredChibiTools.call(
                tool_name=str(function_call.name),
                tools_args=tool_context | copy(function_call.args) if function_call.args else tool_context,
            )
            for function_call in response.function_calls
        ]
        results = await asyncio.gather(*tool_coroutines)

        thought_signature = self._get_thought_signature(response=response)
        if not thought_signature:
            logger.error(
                f"Could not get thought signature for function call, no response candidates found: "
                f"{response.candidates}."
            )

        for function_call, result in zip(response.function_calls, results):
            function_call_id = function_call.id or str(uuid4())
            tool_call_message: ContentDict = ContentDict(
                role="model",
                parts=[
                    PartDict(
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
                            id=function_call_id, name=function_call.name, response=result.model_dump()
                        )
                    ),
                ],
            )

            messages.append(tool_call_message)
            messages.append(tool_result_message)

        logger.log("CALL", "All the function results have been obtained. Returning them to the LLM...")
        return await self._get_chat_completion_response(
            messages=messages,
            model=model_name,
            user=user,
            system_prompt=system_prompt,
            context=context,
            update=update,
        )

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
            messages=initial_messages.copy(),
            user=user,
            model=model,
            system_prompt=system_prompt,
            context=context,
            update=update,
        )

        new_messages = [msg for msg in updated_messages if msg not in initial_messages]
        return chat_response, [Message.from_google(msg) for msg in new_messages]

    async def _generate_image_via_content_creation_model(
        self,
        prompt: str,
        model: str,
    ) -> list[Image]:
        image_size = (
            gpt_settings.image_size_nano_banana if "flash" not in model else None
        )  # flash-models don't support it

        http_options = HttpOptions(httpx_async_client=self.get_async_httpx_client())

        generation_config = GenerateContentConfig(
            image_config=ImageConfig(
                aspect_ratio=gpt_settings.image_aspect_ratio,
                image_size=image_size,
            )
        )

        async with Client(api_key=gpt_settings.gemini_key, http_options=http_options).aio as client:
            response: GenerateContentResponse = await client.models.generate_content(
                model=model,
                contents=[prompt],
                config=generation_config,
            )
        if not response.parts:
            raise ServiceResponseError(provider=self.name, model=model, detail="No content-parts in response found")

        images: list[Image | None] = [part.as_image() for part in response.parts if part]
        return [image for image in images if image]

    async def _generate_image_by_imagen(
        self,
        prompt: str,
        model: str,
    ) -> list[Image]:
        http_options = HttpOptions(httpx_async_client=self.get_async_httpx_client())

        if "preview" in model or "fast" in model:
            image_size = None
        else:
            image_size = gpt_settings.image_size_imagen

        generation_config = GenerateImagesConfig(
            aspect_ratio=gpt_settings.image_aspect_ratio,
            number_of_images=gpt_settings.image_n_choices,
            http_options=http_options,
            image_size=image_size,
        )
        async with Client(api_key=gpt_settings.gemini_key).aio as client:
            response: GenerateImagesResponse = await client.models.generate_images(
                model=model,
                prompt=prompt,
                config=generation_config,
            )
            images_in_response = response.images

            return [image for image in images_in_response if image]

    async def get_images(self, prompt: str, model: str | None = None) -> list[BytesIO]:
        selected_model = model or self.default_image_model

        if "imagen-" in selected_model:
            images = await self._generate_image_by_imagen(prompt=prompt, model=selected_model)
        else:
            images = await self._generate_image_via_content_creation_model(prompt=prompt, model=selected_model)

        return [BytesIO(image.image_bytes) for image in images if image.image_bytes]

    @classmethod
    def is_image_ready_model(cls, model_name: str) -> bool:
        return "image" in model_name

    def get_model_display_name(self, model_name: str) -> str:
        if "gemini-3-pro-image" in model_name:
            display_name = model_name.replace("models/gemini-3-pro-image", "Nano Banana Pro")
            return display_name.replace("-", " ").capitalize()

        if "gemini-2.5-flash" in model_name:
            display_name = model_name.replace("models/gemini-2.5-flash-image", "Nano Banana")
            return display_name.replace("-", " ").capitalize()

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
