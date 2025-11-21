import inspect
import json
from typing import Any, Callable, Coroutine, Type, TypeVar

from anthropic.types import (
    Message as AnthropicMessage,
)
from google.genai.types import GenerateContentResponse
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion
from telegram import Update, constants
from telegram.ext import ContextTypes

from chibi.config import gpt_settings
from chibi.models import User
from chibi.schemas.app import UsageSchema

T = TypeVar("T")
M = TypeVar("M", bound=Callable[..., Coroutine[Any, Any, Any]])


def decorate_async_methods(decorator: Callable[[M], M]) -> Callable[[Type[T]], Type[T]]:
    def decorate(cls: Type[T]) -> Type[T]:
        for attr in cls.__dict__:
            if inspect.iscoroutinefunction(getattr(cls, attr)):
                original_func = getattr(cls, attr)
                decorated_func = decorator(original_func)
                setattr(cls, attr, decorated_func)
        return cls

    return decorate


def escape_and_truncate(message: str | dict[str, Any] | None, limit: int = 80) -> str:
    if not message:
        return "no data"
    text = json.dumps(message) if isinstance(message, dict) else message

    escaped_message = text.replace("<", r"\<").replace(">", r"\>")

    if len(escaped_message) < limit + 20:
        return escaped_message
    return f"{escaped_message[:limit]}... (truncated)"


async def prepare_system_prompt(base_system_prompt: str, user: User) -> str:
    prompt = {
        "user_id": user.id,
        "user_info": user.info,
        "system_prompt": base_system_prompt,
    }
    return json.dumps(prompt)


async def send_llm_thoughts(thoughts: str, update: Update | None, context: ContextTypes.DEFAULT_TYPE | None) -> None:
    if not gpt_settings.show_llm_thoughts:
        return None

    from chibi.utils import send_long_message

    if update is None or context is None:
        return None
    message = f"💡💭 {thoughts}"
    await send_long_message(
        message=message,
        update=update,
        context=context,
        parse_mode=constants.ParseMode.MARKDOWN_V2,
        reply=False,
    )


def get_usage_from_anthropic_response(response_message: AnthropicMessage) -> UsageSchema:
    return UsageSchema(
        completion_tokens=response_message.usage.output_tokens,
        prompt_tokens=response_message.usage.input_tokens,
        cache_creation_input_tokens=response_message.usage.cache_creation_input_tokens,
        cache_read_input_tokens=response_message.usage.cache_read_input_tokens,
        total_tokens=response_message.usage.output_tokens + response_message.usage.input_tokens,
    )


def get_usage_from_openai_response(response_message: ChatCompletion) -> UsageSchema:
    if response_message.usage is None:
        return UsageSchema()
    response_usage = response_message.usage
    usage = UsageSchema(
        completion_tokens=response_usage.completion_tokens,
        prompt_tokens=response_usage.prompt_tokens,
        total_tokens=response_usage.total_tokens,
    )
    if prompt_cache := response_usage.prompt_tokens_details:
        usage.cache_read_input_tokens = prompt_cache.cached_tokens or 0
    return usage


def get_usage_from_google_response(response_message: GenerateContentResponse) -> UsageSchema:
    if not response_message.usage_metadata:
        return UsageSchema()

    return UsageSchema(
        total_tokens=response_message.usage_metadata.total_token_count or 0,
        completion_tokens=response_message.usage_metadata.candidates_token_count or 0,
        prompt_tokens=response_message.usage_metadata.prompt_token_count or 0,
        cache_read_input_tokens=response_message.usage_metadata.cached_content_token_count or 0,
    )


def get_usage_msg(usage: UsageSchema | CompletionUsage | None) -> str:
    if usage is None:
        return ""
    cache_read = getattr(usage, "cache_read_input_tokens", None)
    cache_create = getattr(usage, "cache_creation_input_tokens", None)
    return (
        f"Tokens used: {getattr(usage, 'total_tokens', None) or 'n/a'} "
        f"({getattr(usage, 'prompt_tokens', None)} prompt, "
        f"{getattr(usage, 'completion_tokens', None)} completion, "
        f"{cache_read or 0} cached read/prompt, "
        f"{cache_create or 0} cached creation)"
    )
