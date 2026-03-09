import inspect
import json
import os
import platform
from typing import Any, Callable, Coroutine, ParamSpec, Type, TypeAlias, TypeVar

from anthropic.types import (
    Message as AnthropicMessage,
)
from google.genai.types import GenerateContentResponse
from mistralai import ChatCompletionResponse
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion

from chibi.config import application_settings, gpt_settings
from chibi.models import User
from chibi.schemas.app import UsageSchema
from chibi.schemas.suno import SunoGetGenerationDetailsSchema
from chibi.services.interface import UserInterface
from chibi.storage.files import get_file_storage
from chibi.storage.files.file_storage import FileStorage
from chibi.utils.app import get_builtin_skill_names

T = TypeVar("T")
P = ParamSpec("P")
M = TypeVar("M", bound=Callable[..., Coroutine[Any, Any, Any]])
AsyncFunc: TypeAlias = Callable[P, Coroutine[Any, Any, T]]


def decorate_async_methods(decorator: Callable[[M], M]) -> Callable[[Type[T]], Type[T]]:
    def decorate(cls: Type[T]) -> Type[T]:
        for attr in cls.__dict__:
            if inspect.iscoroutinefunction(getattr(cls, attr)):
                original_func = getattr(cls, attr)
                decorated_func = decorator(original_func)
                setattr(cls, attr, decorated_func)
        return cls

    return decorate


def escape_and_truncate(message: str | dict[str, Any] | list[dict[str, Any]] | None, limit: int = 50) -> str:
    if not message:
        return "no data"

    if isinstance(message, dict):
        return json.dumps({k: escape_and_truncate(message=v, limit=limit) for k, v in message.items()})

    if isinstance(message, list):
        return json.dumps([escape_and_truncate(message=m, limit=limit) for m in message])

    escaped_message = str(message).replace("<", r"\<").replace(">", r"\>")
    if len(escaped_message) < limit + 20:
        return escaped_message
    return f"{escaped_message[:limit]}... (truncated)"


async def prepare_system_prompt(base_system_prompt: str, user: User, interface: UserInterface | None) -> str:
    prompt: dict[str, Any] = {
        "system_prompt": base_system_prompt,
        "available_builtin_skills": get_builtin_skill_names(),
    }

    if gpt_settings.filesystem_access:
        system_data = {
            "current_working_dir": user.working_dir,
            "platform": platform.platform(),
            "shell": os.environ.get("SHELL", "unknown"),
            "running_inside_container": application_settings.running_in_container,
        }
        if application_settings.running_in_container:
            system_data["container_type"] = application_settings.runtime_environment

        prompt["system"] = system_data

    if interface:
        storage: FileStorage = get_file_storage(interface=interface)
        prompt["last_uploaded_files"] = await storage.get_available_files(limit=10)

    prompt.update({"user_id": user.id, "user_info": user.info, "activated_skills": user.llm_skills})
    return json.dumps(prompt)


async def send_llm_thoughts(thoughts: str, interface: UserInterface | None = None) -> None:
    if not gpt_settings.show_llm_thoughts:
        return None

    if not interface:
        return None

    message = f"💡💭 {thoughts}"

    await interface.send_message(message=message, reply=False)


def get_usage_from_anthropic_response(response_message: AnthropicMessage) -> UsageSchema:
    return UsageSchema(
        completion_tokens=response_message.usage.output_tokens,
        prompt_tokens=response_message.usage.input_tokens,
        cache_creation_input_tokens=response_message.usage.cache_creation_input_tokens or 0,
        cache_read_input_tokens=response_message.usage.cache_read_input_tokens or 0,
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


def get_usage_from_mistral_response(response_message: ChatCompletionResponse) -> UsageSchema:
    return UsageSchema(
        completion_tokens=response_message.usage.completion_tokens or 0,
        prompt_tokens=response_message.usage.prompt_tokens or 0,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        total_tokens=response_message.usage.total_tokens or 0,
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


def suno_task_still_processing(task_data_response: SunoGetGenerationDetailsSchema) -> bool:
    return task_data_response.is_in_progress


# def limit_recursion(
#     max_depth: int = application_settings.max_consecutive_tool_calls,
# ) -> Callable[[AsyncFunc[P, T]], AsyncFunc[P, T]]:
#     def decorator(func: AsyncFunc[P, T]) -> AsyncFunc[P, T]:
#         depth_var: ContextVar[int] = ContextVar(f"{func.__name__}_depth", default=0)
#
#         @wraps(func)
#         async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
#             current_depth = depth_var.get()
#             depth_var.set(current_depth + 1)
#             if depth_var.get() > max_depth + 1:
#                 depth_var.set(current_depth)
#                 class_name = ""
#                 if args and hasattr(args[0], "__class__"):
#                     class_name = f"{args[0].__class__.__name__}."
#                 raise RecursionLimitExceeded(
#                     provider=class_name,
#                     model=cast(str, kwargs.get("model", "unknown")),
#                     detail=f"Recursion depth exceeded: {max_depth} (function: {class_name}{func.__name__})",
#                     exceeded_limit=max_depth,
#                 )
#
#             try:
#                 result = await func(*args, **kwargs)
#                 return result
#             finally:
#                 depth_var.set(current_depth)
#
#         return async_wrapper
#
#     return decorator
