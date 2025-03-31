import inspect
from abc import ABC
from functools import wraps
from io import BytesIO
from typing import (
    Awaitable,
    Callable,
    Generic,
    Iterable,
    Literal,
    ParamSpec,
    TypeVar,
    cast,
)

import httpx
from httpx import Response
from httpx._types import QueryParamTypes, RequestData
from loguru import logger
from openai import (
    APIConnectionError,
    AsyncOpenAI,
    AuthenticationError,
    NotGiven,
    OpenAIError,
    RateLimitError,
)
from openai.types import ImagesResponse
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion import Choice

from chibi.config import gpt_settings
from chibi.constants import IMAGE_SIZE_LITERAL
from chibi.exceptions import (
    NoApiKeyProvidedError,
    NoModelSelectedError,
    NotAuthorizedError,
    ServiceConnectionError,
    ServiceRateLimitError,
    ServiceResponseError,
)
from chibi.schemas.app import ChatResponseSchema
from chibi.types import ChatCompletionMessageSchema

P = ParamSpec("P")
R = TypeVar("R")


class Provider(ABC):
    name: str
    model_name_keywords: list[str] = []
    model_name_prefixes: list[str] = []
    model_name_keywords_exclude: list[str] = []
    default_model: str
    default_image_model: str | None = None
    temperature: float | NotGiven | None = gpt_settings.temperature
    max_tokens: int | NotGiven | None = gpt_settings.max_tokens
    presence_penalty: float | NotGiven | None = gpt_settings.presence_penalty
    frequency_penalty: float | NotGiven | None = gpt_settings.frequency_penalty
    timeout: int = gpt_settings.timeout
    image_quality: Literal["standard", "hd"] | NotGiven = gpt_settings.image_quality
    image_size: IMAGE_SIZE_LITERAL | NotGiven | None = gpt_settings.image_size

    def __init__(self, token: str, user=None) -> None:
        self.token = token
        # self.active_model = user.model if user else None

    async def _get_chat_completion_response(
        self,
        messages: list[ChatCompletionMessageSchema],
        model: str,
        system_prompt: str,
    ) -> ChatResponseSchema:
        raise NotImplementedError

    async def get_chat_response(
        self,
        messages: list[ChatCompletionMessageSchema],
        model: str | None = None,
        system_prompt: str = gpt_settings.assistant_prompt,
    ) -> ChatResponseSchema:
        model = model or self.default_model
        return await self._get_chat_completion_response(
            model=model,
            messages=messages,
            system_prompt=system_prompt,
        )

    async def get_available_models(self, image_generation: bool = False) -> list[str]:
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

    async def _get_chat_completion_response(
        self,
        messages: list[ChatCompletionMessageSchema],
        model: str,
        system_prompt: str,
    ) -> ChatResponseSchema:
        system_message = ChatCompletionSystemMessageParam(role="system", content=system_prompt)
        dialog: Iterable[ChatCompletionMessageParam] = [system_message] + messages  # type: ignore

        response = await self.client.chat.completions.create(
            model=model,
            messages=dialog,
            temperature=self.temperature,
            max_completion_tokens=self.max_tokens,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            timeout=self.timeout,
        )
        if len(response.choices) != 0:
            choices: list[Choice] = response.choices
            data = choices[0]
            answer = data.message.content
            usage = response.usage
        else:
            answer = ""
            usage = None

        return ChatResponseSchema(answer=answer or "", provider=self.name, model=model, usage=usage)

    async def get_available_models(self, image_generation: bool = False) -> list[str]:
        models = await self.client.models.list()
        if image_generation:
            image_ready_model_names = [model.id for model in models.data if self.is_image_ready_model(model.id)]
            return sorted([model for model in image_ready_model_names])

        if gpt_settings.models_whitelist:
            allowed_model_names = [model.id for model in models.data if model.id in gpt_settings.models_whitelist]
        else:
            allowed_model_names = [model.id for model in models.data if self.is_chat_ready_model(model.id)]
        return sorted([model for model in allowed_model_names])

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
        return [image.url for image in response.data if image.url]


class RestApiFriendlyProvider(Provider):
    @property
    def _headers(self) -> dict[str, str]:
        raise NotImplementedError

    async def _request(
        self, method: str, url: str, data: RequestData | None = None, params: QueryParamTypes | None = None
    ) -> Response:
        if not self.token:
            raise NoApiKeyProvidedError(provider=self.name)

        transport = httpx.AsyncHTTPTransport(retries=gpt_settings.retries, proxy=gpt_settings.proxy)

        try:
            async with httpx.AsyncClient(
                transport=transport, timeout=gpt_settings.timeout, proxy=gpt_settings.proxy
            ) as client:
                response = await client.request(method=method, url=url, json=data, headers=self._headers, params=params)
        except Exception as e:
            logger.error(f"An error occurred while calling the {self.name} API: {e}")
            raise ServiceResponseError(provider=self.name, detail=str(e))

        if response.status_code == 200:
            return response

        logger.error(
            f"Unexpected response from {self.name} API. Status code: {response.status_code}. " f"Data: {response.text}"
        )
        if response.status_code == 401:
            raise NotAuthorizedError(provider=self.name)
        if response.status_code == 429:
            raise ServiceRateLimitError(provider=self.name)
        raise ServiceResponseError(provider=self.name)
