import asyncio
import inspect
import json
import random
from abc import ABC
from asyncio import sleep
from functools import wraps
from io import BytesIO
from typing import Any, Awaitable, Callable, Generic, Literal, Optional, ParamSpec, TypeVar, cast

import httpx
from anthropic import AsyncClient, NotGiven, Omit
from anthropic.types import (
    CacheControlEphemeralParam,
    MessageParam,
    TextBlock,
    TextBlockParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlock,
)
from anthropic.types import (
    Message as AnthropicMessage,
)
from httpx import Response
from httpx._types import QueryParamTypes, RequestData
from loguru import logger
from openai import (
    NOT_GIVEN,
    APIConnectionError,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)
from openai import (
    NotGiven as OpenAINotGiven,
)
from openai.types import ImagesResponse
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
)
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message_tool_call_param import Function
from telegram import Update
from telegram.ext import ContextTypes

from chibi.config import application_settings, gpt_settings
from chibi.constants import IMAGE_SIZE_LITERAL
from chibi.exceptions import (
    NoApiKeyProvidedError,
    NoModelSelectedError,
    NoResponseError,
    NotAuthorizedError,
    ServiceConnectionError,
    ServiceRateLimitError,
    ServiceResponseError,
)
from chibi.models import Message, User
from chibi.schemas.app import ChatResponseSchema, ModelChangeSchema, ModeratorsAnswer
from chibi.services.metrics import MetricsService
from chibi.services.providers.tools import RegisteredChibiTools
from chibi.services.providers.tools.constants import MODERATOR_PROMPT
from chibi.services.providers.utils import (
    get_usage_from_anthropic_response,
    get_usage_from_openai_response,
    get_usage_msg,
    prepare_system_prompt,
    send_llm_thoughts,
)

P = ParamSpec("P")
R = TypeVar("R")


class RegisteredProviders:
    all: dict[str, type["Provider"]] = {}
    available: dict[str, type["Provider"]] = {}

    def __init__(self, user_api_keys: dict[str, str] | None = None) -> None:
        self.tokens = {} if not user_api_keys else user_api_keys
        if gpt_settings.public_mode:
            self.available: dict[str, type["Provider"]] = {
                provider.name: provider for provider in RegisteredProviders.all.values() if provider.name in self.tokens
            }

    @classmethod
    def register(cls, provider: type["Provider"]) -> None:
        cls.all[provider.name] = provider

    @classmethod
    def register_as_available(cls, provider: type["Provider"]) -> None:
        cls.available[provider.name] = provider

    def get_api_key(self, provider: type["Provider"]) -> str | None:
        if not gpt_settings.public_mode:
            return provider.api_key

        if provider.name not in self.tokens:
            return None

        return self.tokens[provider.name]

    @property
    def available_instances(self) -> list["Provider"]:
        return [
            provider(token=self.get_api_key(provider))  # type: ignore
            for provider in self.available.values()
            if self.get_api_key(provider) is not None
        ]

    @property
    def chat_ready(self) -> dict[str, type["Provider"]]:
        return {provider_name: provider for provider_name, provider in self.available.items() if provider.chat_ready}

    @property
    def moderation_ready(self) -> dict[str, type["Provider"]]:
        return {
            provider_name: provider for provider_name, provider in self.available.items() if provider.moderation_ready
        }

    @property
    def image_generation_ready(self) -> dict[str, type["Provider"]]:
        return {name: provider for name, provider in self.available.items() if provider.image_generation_ready}

    @property
    def stt_ready(self) -> dict[str, type["Provider"]]:
        return {name: provider for name, provider in self.available.items() if provider.stt_ready}

    @property
    def tts_ready(self) -> dict[str, type["Provider"]]:
        return {name: provider for name, provider in self.available.items() if provider.tts_ready}

    def get_instance(self, provider: type["Provider"]) -> Optional["Provider"]:
        api_key = self.get_api_key(provider)
        if not api_key:
            return None
        return provider(token=api_key)

    def get(self, provider_name: str) -> Optional["Provider"]:
        if provider_name not in self.available:
            return None
        provider = self.available[provider_name]
        return self.get_instance(provider=provider)

    @classmethod
    def get_class(cls, provider_name: str) -> Optional[type["Provider"]]:
        return cls.all.get(provider_name)

    @property
    def first_tts_ready(self) -> Optional["Provider"]:
        if provider := next(iter(self.tts_ready.values()), None):
            return self.get_instance(provider=provider)
        return None

    @property
    def first_stt_ready(self) -> Optional["Provider"]:
        if provider := next(iter(self.stt_ready.values()), None):
            return self.get_instance(provider=provider)
        return None

    @property
    def first_image_generation_ready(self) -> Optional["Provider"]:
        if provider := next(iter(self.image_generation_ready.values()), None):
            return self.get_instance(provider=provider)
        return None

    @property
    def first_chat_ready(self) -> Optional["Provider"]:
        if provider := next(iter(self.chat_ready.values()), None):
            return self.get_instance(provider=provider)
        return None

    @property
    def first_moderation_ready(self) -> Optional["Provider"]:
        if provider := next(reversed(self.moderation_ready.values()), None):
            return self.get_instance(provider=provider)
        return None


class Provider(ABC):
    api_key: str | None = None
    stt_ready: bool = False
    tts_ready: bool = False
    ocr_ready: bool = False
    chat_ready: bool = False
    moderation_ready: bool = False
    image_generation_ready: bool = False

    name: str
    model_name_keywords: list[str] = []
    model_name_prefixes: list[str] = []
    model_name_keywords_exclude: list[str] = []
    default_model: str
    default_image_model: str | None = None
    default_stt_model: str | None = None
    default_tts_voice: str | None = None
    default_tts_model: str | None = None
    default_moderation_model: str | None = None
    timeout: int = gpt_settings.timeout

    def __init__(self, token: str) -> None:
        self.token = token

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if hasattr(cls, "name"):
            RegisteredProviders.register(cls)

        if cls.api_key:
            RegisteredProviders.register_as_available(cls)

    async def get_chat_response(
        self,
        messages: list[Message],
        user: User | None = None,
        model: str | None = None,
        system_prompt: str = gpt_settings.assistant_prompt,
        update: Update | None = None,
        context: ContextTypes.DEFAULT_TYPE | None = None,
    ) -> tuple[ChatResponseSchema, list[Message]]:
        raise NotImplementedError

    async def get_available_models(self, image_generation: bool = False) -> list[ModelChangeSchema]:
        raise NotImplementedError

    def get_model_display_name(self, model_name: str) -> str:
        raise NotImplementedError

    async def transcribe(self, audio: BytesIO, model: str | None = None) -> str:
        raise NotImplementedError

    async def speech(self, text: str, voice: str | None = None, model: str | None = None) -> bytes:
        raise NotImplementedError

    async def moderate_command(self, cmd: str, model: str | None = None) -> ModeratorsAnswer:
        raise NotImplementedError

    async def api_key_is_valid(self) -> bool:
        try:
            await self.get_available_models()
        except Exception:  # Some providers return 403, others - 400... Okay..
            return False
        return True

    @classmethod
    def _model_name_has_prefix(cls, model_name: str) -> bool:
        if not cls.model_name_prefixes:
            return True
        for prefix in cls.model_name_prefixes:
            if model_name.startswith(prefix):
                return True
        return False

    @classmethod
    def _model_name_has_keyword(cls, model_name: str) -> bool:
        if not cls.model_name_keywords:
            return True
        for keyword in cls.model_name_keywords:
            if keyword in model_name:
                return True
        return False

    @classmethod
    def _model_name_has_keywords_exclude(cls, model_name: str) -> bool:
        if not cls.model_name_keywords_exclude:
            return False
        for keyword in cls.model_name_keywords_exclude:
            if keyword in model_name:
                return True
        return False

    @classmethod
    def is_chat_ready_model(cls, model_name: str) -> bool:
        return all(
            (
                cls._model_name_has_prefix(model_name),
                cls._model_name_has_keyword(model_name),
                not cls._model_name_has_keywords_exclude(model_name),
            )
        )

    @classmethod
    def is_image_ready_model(cls, model_name: str) -> bool:
        return "image" in model_name

    async def get_images(self, prompt: str, model: str | None) -> list[str] | list[BytesIO]:
        raise NotImplementedError


class OpenAIFriendlyProvider(Provider, Generic[P, R]):
    temperature: float | OpenAINotGiven | None = gpt_settings.temperature
    max_tokens: int | OpenAINotGiven | None = gpt_settings.max_tokens
    presence_penalty: float | OpenAINotGiven | None = gpt_settings.presence_penalty
    frequency_penalty: float | OpenAINotGiven | None = gpt_settings.frequency_penalty
    image_quality: Literal["standard", "hd"] | OpenAINotGiven = gpt_settings.image_quality
    image_size: IMAGE_SIZE_LITERAL | OpenAINotGiven | None = gpt_settings.image_size
    base_url: str
    image_n_choices: int = gpt_settings.image_n_choices

    def __getattribute__(self, name: str) -> object:
        attr = super().__getattribute__(name)

        if callable(attr):
            if inspect.iscoroutinefunction(attr):
                attr_async_callable = cast(Callable[P, Awaitable[R]], attr)

                @wraps(attr_async_callable)
                async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R | None:
                    model_name = cast(str, kwargs.get("model", "unknown"))
                    try:
                        return await attr_async_callable(*args, **kwargs)
                    except APIConnectionError:
                        raise ServiceConnectionError(provider=self.name, model=model_name)
                    except AuthenticationError:
                        raise NotAuthorizedError(provider=self.name, model=model_name)
                    except RateLimitError:
                        raise ServiceRateLimitError(provider=self.name, model=model_name)
                    except OpenAIError as e:
                        logger.error(e)
                        raise ServiceResponseError(provider=self.name, model=model_name)

                return async_wrapper
            else:
                attr_callable = cast(Callable[P, R], attr)

                @wraps(attr_callable)
                def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R | None:
                    return attr_callable(*args, **kwargs)

                return sync_wrapper

        return attr

    @property
    def client(self) -> AsyncOpenAI:
        if not self.token:
            raise NoApiKeyProvidedError(provider=self.name)
        return AsyncOpenAI(api_key=self.token, base_url=self.base_url)

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

        initial_messages = [msg.to_openai() for msg in messages]
        chat_response, updated_messages = await self._get_chat_completion_response(
            messages=initial_messages.copy(),
            model=model,
            system_prompt=system_prompt,
            user=user,
            context=context,
            update=update,
        )
        new_messages = [msg for msg in updated_messages if msg not in initial_messages]
        return (
            chat_response,
            [Message.from_openai(msg) for msg in new_messages],
        )

    async def _get_chat_completion_response(
        self,
        messages: list[ChatCompletionMessageParam],
        model: str,
        user: User | None = None,
        system_prompt: str | None = None,
        update: Update | None = None,
        context: ContextTypes.DEFAULT_TYPE | None = None,
    ) -> tuple[ChatResponseSchema, list[ChatCompletionMessageParam]]:
        if not system_prompt:
            dialog = messages
        else:
            prepared_system_prompt = await prepare_system_prompt(base_system_prompt=system_prompt, user=user)
            system_message = ChatCompletionSystemMessageParam(role="system", content=prepared_system_prompt)
            dialog: list[ChatCompletionMessageParam] = [system_message] + messages  # type: ignore

        temperature = 1 if model.startswith("o") else self.temperature

        response: ChatCompletion = await self.client.chat.completions.create(
            model=model,
            messages=dialog,
            temperature=temperature,
            max_tokens=self.max_tokens,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            timeout=self.timeout,
            tools=RegisteredChibiTools.get_tool_definitions(),
            tool_choice="auto",
            reasoning_effort="medium" if "reason" in model else NOT_GIVEN,
        )
        choices: list[Choice] = response.choices

        if len(choices) == 0:
            raise ServiceResponseError(provider=self.name, model=model, detail="Unexpected (empty) response received")

        data = choices[0]
        answer: str = data.message.content or ""

        usage = get_usage_from_openai_response(response_message=response)
        if application_settings.is_influx_configured:
            MetricsService.send_usage_metrics(metric=usage, model=model, provider=self.name, user=user)
        usage_message = get_usage_msg(usage=usage)

        tool_calls: list[ChatCompletionMessageToolCall] | None = data.message.tool_calls

        if not tool_calls:
            messages.append(ChatCompletionAssistantMessageParam(**data.message.model_dump()))  # type: ignore
            return ChatResponseSchema(answer=answer, provider=self.name, model=model, usage=usage), messages

        # Tool calls handling
        logger.log("CALL", f"{model} requested the call of {len(tool_calls)} tools.")

        thoughts = answer or "No thoughts"
        if answer:
            await send_llm_thoughts(thoughts=thoughts, context=context, update=update)
        logger.log("THINK", f"{model}: {thoughts}. {usage_message}")

        tool_context: dict[str, Any] = {
            "user_id": user.id if user else None,
            "telegram_context": context,
            "telegram_update": update,
            "model": model,
        }

        tool_coroutines = [
            RegisteredChibiTools.call(
                tool_name=tool_call.function.name, tools_args=tool_context | json.loads(tool_call.function.arguments)
            )
            for tool_call in tool_calls
        ]
        results = await asyncio.gather(*tool_coroutines)

        for tool_call, result in zip(tool_calls, results):
            tool_call_message = ChatCompletionAssistantMessageParam(
                role="assistant",
                content=answer,
                tool_calls=[
                    ChatCompletionMessageToolCallParam(
                        id=tool_call.id,
                        type="function",
                        function=Function(
                            name=tool_call.function.name,
                            arguments=tool_call.function.arguments,
                        ),
                    )
                ],
            )
            tool_result_message = ChatCompletionToolMessageParam(
                tool_call_id=tool_call.id,
                role="tool",
                content=result.model_dump_json(),
            )
            messages.append(tool_call_message)
            messages.append(tool_result_message)

        logger.log("CALL", "All the function results have been obtained. Returning them to the LLM...")
        return await self._get_chat_completion_response(
            messages=messages,
            model=model,
            user=user,
            system_prompt=system_prompt,
            context=context,
            update=update,
        )

    async def moderate_command(self, cmd: str, model: str | None = None) -> ModeratorsAnswer:
        moderator_model = model or self.default_moderation_model or self.default_model
        system_message = ChatCompletionSystemMessageParam(role="system", content=MODERATOR_PROMPT)

        messages = [
            Message(role="user", content=cmd).to_openai(),
        ]

        dialog: list[ChatCompletionMessageParam] = [system_message] + messages
        temperature = (
            1 if moderator_model.startswith("o") or "mini" in moderator_model or "nano" in moderator_model else 0.0
        )
        response: ChatCompletion = await self.client.chat.completions.create(
            model=moderator_model,
            messages=dialog,
            temperature=temperature,
            max_completion_tokens=1024,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            timeout=self.timeout,
            # reasoning_effort="medium" if "reason" in moderator_model else NOT_GIVEN,
        )

        choices: list[Choice] = response.choices

        if len(choices) == 0:
            raise ServiceResponseError(
                provider=self.name, model=moderator_model, detail="Unexpected (empty) response received"
            )

        data = choices[0]
        answer: str = data.message.content or ""

        usage = get_usage_from_openai_response(response_message=response)
        if application_settings.is_influx_configured:
            MetricsService.send_usage_metrics(metric=usage, model=moderator_model, provider=self.name)
        # usage_message = get_usage_msg(usage=usage)
        answer = answer.strip("```")
        answer = answer.strip("json")
        answer = answer.strip()
        try:
            result_data = json.loads(answer)
        except Exception:
            logger.error(f"Error parsing moderator's response: {answer}")
            return ModeratorsAnswer(verdict="declined", reason=answer, status="error")

        verdict = result_data.get("verdict", "declined")
        if verdict == "accepted":
            return ModeratorsAnswer(verdict="accepted", status="ok")

        reason = result_data.get("reason", None)
        if reason is None:
            logger.error(f"Moderator did not provide reason properly: {answer}")

        return ModeratorsAnswer(verdict="declined", reason=reason, status="operation aborted")

    def get_model_display_name(self, model_name: str) -> str:
        return model_name.replace("-", " ")

    async def get_available_models(self, image_generation: bool = False) -> list[ModelChangeSchema]:
        try:
            models = await self.client.models.list()
        except Exception as e:
            logger.error(f"Failed to get available models for provider {self.name} due to exception: {e}")
            return []

        all_models = [
            ModelChangeSchema(
                provider=self.name,
                name=model.id,
                display_name=self.get_model_display_name(model.id),
                image_generation=self.is_image_ready_model(model.id),
            )
            for model in models.data
        ]
        all_models.sort(key=lambda model: model.name)

        if image_generation:
            return [model for model in all_models if model.image_generation]

        if gpt_settings.models_whitelist:
            return [model for model in all_models if model.name in gpt_settings.models_whitelist]

        return [model for model in all_models if self.is_chat_ready_model(model.name)]

    async def _get_image_generation_response(self, prompt: str, model: str) -> ImagesResponse:
        return await self.client.images.generate(
            model=model,
            prompt=prompt,
            n=gpt_settings.image_n_choices,
            quality=self.image_quality,
            size=self.image_size,
            timeout=gpt_settings.timeout,
            response_format="url",
        )

    async def get_images(self, prompt: str, model: str | None = None) -> list[str] | list[BytesIO]:
        model = model or self.default_image_model
        if not model:
            raise NoModelSelectedError(provider=self.name, detail="No image generation model selected")
        response = await self._get_image_generation_response(prompt=prompt, model=model)
        if not response.data:
            raise ServiceResponseError(provider=self.name, model=model, detail="No image data received.")
        return [image.url for image in response.data if image.url]


class RestApiFriendlyProvider(Provider):
    @property
    def _headers(self) -> dict[str, str]:
        raise NotImplementedError

    def get_async_httpx_client(self) -> httpx.AsyncClient:
        transport = httpx.AsyncHTTPTransport(retries=gpt_settings.retries, proxy=gpt_settings.proxy)
        return httpx.AsyncClient(transport=transport, timeout=gpt_settings.timeout)

    async def _request(
        self,
        method: str,
        url: str,
        data: RequestData | None = None,
        params: QueryParamTypes | None = None,
        headers: dict[str, str] | None = None,
    ) -> Response:
        if not self.token:
            raise NoApiKeyProvidedError(provider=self.name)

        try:
            async with self.get_async_httpx_client() as client:
                response = await client.request(
                    method=method,
                    url=url,
                    json=data,
                    headers=headers or self._headers,
                    params=params,
                )
        except Exception as e:
            logger.error(f"An error occurred while calling the {self.name} API: {e}")
            raise ServiceResponseError(provider=self.name, detail=str(e))

        if response.status_code == 200:
            return response

        logger.error(
            f"Unexpected response from {self.name} API. Status code: {response.status_code}. Data: {response.text}"
        )
        if response.status_code == 401:
            raise NotAuthorizedError(provider=self.name)
        if response.status_code == 429:
            raise ServiceRateLimitError(provider=self.name)
        raise ServiceResponseError(provider=self.name)


class AnthropicFriendlyProvider(RestApiFriendlyProvider):
    frequency_penalty: float | NotGiven | None = gpt_settings.frequency_penalty
    max_tokens: int = gpt_settings.max_tokens
    presence_penalty: float | NotGiven = gpt_settings.presence_penalty
    temperature: float | Omit = gpt_settings.temperature

    @property
    def tools_list(self) -> list[ToolParam]:
        anthropic_tools = [
            ToolParam(
                name=tool["function"]["name"],
                description=tool["function"]["description"],
                input_schema=tool["function"]["parameters"],
            )
            for tool in RegisteredChibiTools.get_tool_definitions()
        ]
        return anthropic_tools

    @property
    def client(self) -> AsyncClient:
        raise NotImplementedError

    async def _generate_content(
        self,
        model: str,
        system_prompt: str,
        messages: list[MessageParam],
    ) -> AnthropicMessage:
        for attempt in range(gpt_settings.retries):
            response_message: AnthropicMessage = await self.client.messages.create(
                model=model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=self.timeout,
                tools=self.tools_list,
                system=[
                    TextBlockParam(
                        text=system_prompt,
                        type="text",
                        cache_control=CacheControlEphemeralParam(type="ephemeral"),
                    )
                ],
                messages=messages,
            )

            if response_message.content and len(response_message.content) > 0:
                return response_message

            delay = gpt_settings.backoff_factor * (2**attempt)
            jitter = delay * random.uniform(0.1, 0.5)
            total_delay = delay + jitter

            logger.warning(
                f"Attempt #{attempt + 1}. Unexpected (empty) response received. Retrying in {total_delay} seconds..."
            )
            await sleep(total_delay)
        raise NoResponseError(provider=self.name, model=model, detail="Unexpected (empty) response received")

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
        initial_messages = [msg.to_anthropic() for msg in messages]

        if len(initial_messages) >= 2:
            initial_messages[-2]["content"][0]["cache_control"] = {"type": "ephemeral"}  # type: ignore

        chat_response, updated_messages = await self._get_chat_completion_response(
            messages=initial_messages.copy(),
            user=user,
            model=model,
            system_prompt=system_prompt,
            context=context,
            update=update,
        )
        new_messages = [msg for msg in updated_messages if msg not in initial_messages]
        return (
            chat_response,
            [Message.from_anthropic(msg) for msg in new_messages],
        )

    async def _get_chat_completion_response(
        self,
        messages: list[MessageParam],
        model: str,
        user: User | None = None,
        system_prompt: str = gpt_settings.assistant_prompt,
        update: Update | None = None,
        context: ContextTypes.DEFAULT_TYPE | None = None,
    ) -> tuple[ChatResponseSchema, list[MessageParam]]:
        prepared_system_prompt = await prepare_system_prompt(base_system_prompt=system_prompt, user=user)
        response_message: AnthropicMessage = await self._generate_content(
            model=model,
            system_prompt=prepared_system_prompt,
            messages=messages,
        )
        usage = get_usage_from_anthropic_response(response_message=response_message)

        if application_settings.is_influx_configured:
            MetricsService.send_usage_metrics(metric=usage, user=user, model=model, provider=self.name)

        tool_call_parts = [part for part in response_message.content if isinstance(part, ToolUseBlock)]
        if not tool_call_parts:
            messages.append(
                MessageParam(
                    role="assistant",
                    content=[content.model_dump() for content in response_message.content],  # type: ignore
                )
            )
            answer = None
            for block in response_message.content:
                if answer := getattr(block, "text", None):
                    break

            return ChatResponseSchema(
                answer=answer or "no data",
                provider=self.name,
                model=model,
                usage=usage,
            ), messages

        # Tool calls handling
        logger.log("CALL", f"{model} requested the call of {len(tool_call_parts)} tools.")
        thoughts_part: TextBlock | None = next(
            (part for part in response_message.content if isinstance(part, TextBlock)), None
        )

        if thoughts_part:
            await send_llm_thoughts(thoughts=thoughts_part.text, context=context, update=update)

        logger.log(
            "THINK", f"{model}: {thoughts_part.text if thoughts_part else 'No thoughts'}. {get_usage_msg(usage=usage)}"
        )

        tool_context: dict[str, Any] = {
            "user_id": user.id if user else None,
            "telegram_context": context,
            "telegram_update": update,
            "model": model,
        }

        tool_coroutines = [
            RegisteredChibiTools.call(tool_name=tool_call_part.name, tools_args=tool_context | tool_call_part.input)
            for tool_call_part in tool_call_parts
        ]
        results = await asyncio.gather(*tool_coroutines)

        for tool_call_part, result in zip(tool_call_parts, results):
            tool_call_message = MessageParam(
                role="assistant",
                content=[part.model_dump() for part in (thoughts_part, tool_call_part) if part is not None],  # type: ignore
            )

            tool_result_message = MessageParam(
                role="user",
                content=[
                    ToolResultBlockParam(
                        type="tool_result",
                        tool_use_id=tool_call_part.id,
                        content=result.model_dump_json(),
                    )
                ],
            )
            messages.append(tool_call_message)
            messages.append(tool_result_message)

        logger.log("CALL", "All the function results have been obtained. Returning them to the LLM...")
        return await self._get_chat_completion_response(
            messages=messages,
            model=model,
            user=user,
            system_prompt=system_prompt,
            context=context,
            update=update,
        )

    async def moderate_command(self, cmd: str, model: str | None = None) -> ModeratorsAnswer:
        moderator_model = model or self.default_moderation_model or self.default_model
        messages = [Message(role="user", content=cmd).to_anthropic()]
        response_message: AnthropicMessage = await self.client.messages.create(
            model=moderator_model,
            max_tokens=1024,
            temperature=0.1,
            timeout=self.timeout,
            system=[
                TextBlockParam(
                    text=MODERATOR_PROMPT,
                    type="text",
                )
            ],
            tools=[
                {
                    "name": "print_moderator_verdict",
                    "description": "Provide moderator's verdict",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "verdict": {"type": "string"},
                            "status": {"type": "string", "default": "ok"},
                            "reason": {"type": "string"},
                        },
                        "required": ["verdict"],
                    },
                }
            ],
            tool_choice={"type": "tool", "name": "print_moderator_verdict"},
            messages=messages,
        )
        if not response_message.content:
            return ModeratorsAnswer(status="error", verdict="declined", reason="no response from moderator received")
        usage = get_usage_from_anthropic_response(response_message=response_message)

        if application_settings.is_influx_configured:
            MetricsService.send_usage_metrics(metric=usage, model=moderator_model, provider=self.name)
        tool_call: ToolUseBlock | None = next(
            part for part in response_message.content if isinstance(part, ToolUseBlock)
        )
        if not tool_call:
            return ModeratorsAnswer(status="error", verdict="declined", reason="no response from moderator received")
        answer = tool_call.input

        try:
            return ModeratorsAnswer.model_validate(answer, extra="ignore")
        except Exception as e:
            msg = f"Error parsing moderator's response: {answer}. Error: {e}"
            logger.error(msg)
            return ModeratorsAnswer(verdict="declined", reason=msg, status="error")

    async def get_available_models(self, image_generation: bool = False) -> list[ModelChangeSchema]:
        if image_generation:
            return []

        url = "https://api.anthropic.com/v1/models"

        try:
            response = await self._request(method="GET", url=url)
        except Exception as e:
            logger.error(f"Failed to get available models for provider {self.name} due to exception: {e}")
            return []

        response_data = response.json().get("data", [])
        all_models = [
            ModelChangeSchema(
                provider=self.name,
                name=model.get("id"),
                display_name=model.get("display_name"),
                image_generation=False,
            )
            for model in response_data
            if model.get("id") and model.get("type") == "model"
        ]
        all_models.sort(key=lambda model: model.name)

        if gpt_settings.models_whitelist:
            return [model for model in all_models if model.name in gpt_settings.models_whitelist]

        return all_models
