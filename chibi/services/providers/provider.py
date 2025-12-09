import asyncio
import inspect
import json
from abc import ABC
from functools import wraps
from io import BytesIO
from typing import Any, Awaitable, Callable, Generic, Literal, Optional, ParamSpec, TypeVar, cast

import httpx
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
    NotAuthorizedError,
    ServiceConnectionError,
    ServiceRateLimitError,
    ServiceResponseError,
)
from chibi.models import Message, User
from chibi.schemas.app import ChatResponseSchema, ModelChangeSchema
from chibi.services.metrics import MetricsService
from chibi.services.providers.tools import RegisteredChibiTools, tools
from chibi.services.providers.utils import (
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
    def image_generation_ready(self) -> dict[str, type["Provider"]]:
        return {name: provider for name, provider in self.available.items() if provider.image_generation_ready}

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


class Provider(ABC):
    api_key: str | None = None
    stt_ready: bool = False
    tts_ready: bool = False
    ocr_ready: bool = False
    chat_ready: bool = False
    image_generation_ready: bool = False

    name: str
    model_name_keywords: list[str] = []
    model_name_prefixes: list[str] = []
    model_name_keywords_exclude: list[str] = []
    default_model: str
    default_image_model: str | None = None
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
        # model = model or self.default_model
        # return await self._get_chat_completion_response(messages=messages, model=model, system_prompt=system_prompt)

    async def get_available_models(self, image_generation: bool = False) -> list[ModelChangeSchema]:
        raise NotImplementedError

    def get_model_display_name(self, model_name: str) -> str:
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

        response: ChatCompletion = await self.client.chat.completions.create(
            model=model,
            messages=dialog,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            timeout=self.timeout,
            tools=tools,
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
        logger.log("CALL", f"LLM requested the call of {len(tool_calls)} tools.")

        thoughts = answer or "No thoughts"
        if thoughts:
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

        logger.log("CALL", "The all functions results have been obtained. Returning them to the LLM...")
        return await self._get_chat_completion_response(
            messages=messages,
            model=model,
            user=user,
            system_prompt=system_prompt,
            context=context,
            update=update,
        )

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

    async def _request(
        self,
        method: str,
        url: str,
        data: RequestData | None = None,
        params: QueryParamTypes | None = None,
    ) -> Response:
        if not self.token:
            raise NoApiKeyProvidedError(provider=self.name)

        transport = httpx.AsyncHTTPTransport(retries=gpt_settings.retries, proxy=gpt_settings.proxy)

        try:
            async with httpx.AsyncClient(
                transport=transport,
                timeout=gpt_settings.timeout,
                proxy=gpt_settings.proxy,
            ) as client:
                response = await client.request(
                    method=method,
                    url=url,
                    json=data,
                    headers=self._headers,
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
