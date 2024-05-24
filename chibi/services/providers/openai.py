from functools import wraps
from typing import Any, Callable, Coroutine, Iterable, Type, TypeVar

from openai import (
    APIConnectionError,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion import Choice

from chibi.config import gpt_settings
from chibi.exceptions import (
    NoApiKeyProvidedError,
    NotAuthorizedError,
    ServiceConnectionError,
    ServiceRateLimitError,
    ServiceResponseError,
)
from chibi.schemas.app import ChatResponseSchema
from chibi.services.providers.provider import Provider
from chibi.types import ChatCompletionMessageSchema

T = TypeVar("T")
M = TypeVar("M", bound=Callable[..., Coroutine[Any, Any, Any]])


def handle_openai_exceptions(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        model = kwargs.get("model", "unknown")
        try:
            return await func(*args, **kwargs)
        except APIConnectionError:
            raise ServiceConnectionError(provider="OpenAI", model=model)
        except AuthenticationError:
            raise NotAuthorizedError(provider="OpenAI", model=model)
        except RateLimitError:
            raise ServiceRateLimitError(provider="OpenAI", model=model)
        except OpenAIError:
            raise ServiceResponseError(provider="OpenAI", model=model)

    return wrapper


def decorate_all_methods(decorator: Callable[[M], M]) -> Callable[[Type[T]], Type[T]]:
    def decorate(cls: Type[T]) -> Type[T]:
        for attr in cls.__dict__:
            if callable(getattr(cls, attr)):
                original_func = getattr(cls, attr)
                decorated_func = decorator(original_func)
                setattr(cls, attr, decorated_func)
        return cls

    return decorate


@decorate_all_methods(handle_openai_exceptions)
class OpenAI(Provider):
    name = "OpenAI"

    @property
    def client(self) -> AsyncOpenAI:
        if not self.token:
            raise NoApiKeyProvidedError(provider=self.name, model="unknown")
        return AsyncOpenAI(api_key=self.token)

    async def _get_chat_completion_response(
        self,
        messages: list[ChatCompletionMessageSchema],
        model: str,
        temperature: float = gpt_settings.temperature,
        max_tokens: int = gpt_settings.max_tokens,
        presence_penalty: float = gpt_settings.presence_penalty,
        frequency_penalty: float = gpt_settings.frequency_penalty,
        system_prompt: str = gpt_settings.assistant_prompt,
        timeout: int = gpt_settings.timeout,
    ) -> ChatResponseSchema:
        system_message = ChatCompletionSystemMessageParam(role="system", content=system_prompt)

        dialog: Iterable[ChatCompletionMessageParam] = [system_message] + messages  # type: ignore

        response = await self.client.chat.completions.create(
            model=model,
            messages=dialog,
            temperature=temperature,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            timeout=timeout,
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

    async def get_available_models(self) -> list[str]:
        openai_models = await self.client.models.list()

        if gpt_settings.models_whitelist:
            allowed_model_names = [
                model.id for model in openai_models.data if model.id in gpt_settings.models_whitelist
            ]
        else:
            allowed_model_names = [model.id for model in openai_models.data]

        return sorted([model for model in allowed_model_names if "gpt" in model])

    async def api_key_is_valid(self) -> bool:
        try:
            await self.get_available_models()
        except NotAuthorizedError:
            return False
        except Exception:
            raise
        return True

    async def get_images(self, prompt: str) -> list[str]:
        response = await self.client.images.generate(
            prompt=prompt,
            quality=gpt_settings.image_quality,
            model=gpt_settings.dall_e_model,
            size=gpt_settings.image_size,
            n=gpt_settings.image_n_choices,
        )
        return [image.url for image in response.data if image.url]
