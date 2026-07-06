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
from openai.types.responses import Response
from pydantic import BaseModel, Field

from chibi.config import application_settings, gpt_settings
from chibi.models import Message
from chibi.schemas.app import UsageSchema
from chibi.schemas.suno import SunoGetGenerationDetailsSchema
from chibi.services.interface import UserInterface
from chibi.services.user import get_chibi_user
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


async def prepare_system_prompt(base_system_prompt: str, user_id: int, interface: UserInterface | None) -> str:
    user = await get_chibi_user(user_id=user_id)
    prompt: dict[str, Any] = {
        "system_prompt": base_system_prompt,
        "available_builtin_skills": get_builtin_skill_names(),
    }

    if application_settings.is_chroma_configured:
        retention_days = application_settings.chroma_history_retention_days
        prompt["system_prompt"] += f"""\n\n# Persistent Memory\n
            You can access conversation history from the last {retention_days} days using the
            `search_in_conversation_history` tool.\n\n
            Use memory search when:\n
            - The user refers to past conversations not present in the current context\n
            - The user asks what they said or discussed earlier\n
            - Previous preferences, decisions, or project context may improve the response\n\n
            Guidelines:\n
            - Prefer semantic descriptions over exact quotes when searching\n
            - Do not search memory if the current context already contains the needed information\n
            - Never fabricate recalled information\n
            - If memory results are ambiguous or empty, state that clearly\n
            - Distinguish recalled facts from inferred assumptions\n"""

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

        thread_id = interface.thread_id
        context_size = user.approximate_context_size(thread_id=thread_id)
        prompt["approximate_context_size"] = context_size
        if context_size > gpt_settings.max_history_tokens * 0.7:
            prompt["context_size_warning"] = (
                f"The context size is more than 70% of the maximum allowed ({gpt_settings.max_history_tokens}) tokens. "
                f"It is strongly recommended to reduce the context by calling 'summarize_history' "
                f"or 'clear_tool_call_history' and generating the most detailed summary possible."
            )

    prompt.update({"user_id": user.id, "user_info": user.info, "activated_skills": user.llm_skills})
    return json.dumps(prompt)


class SupervisorToolCallAction(BaseModel):
    """Represents a tool call that the Supervisor must evaluate.

    Used as the ``tool_call`` argument of :func:`build_supervisor_context`
    to describe the action under review in the tool-call supervision
    scenario (as opposed to the final-answer scenario).

    Attributes:
        tool_name: Name of the tool the agent wants to invoke.
        args: Arguments the agent wants to pass to the tool, as a JSON-serializable mapping.
    """

    tool_name: str = Field(description="Name of the tool the agent wants to invoke.")
    args: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments the agent wants to pass to the tool, as a JSON-serializable mapping.",
    )


def _serialize_message_for_supervisor(message: Message) -> dict[str, Any]:
    """Convert a canonical ``Message`` into a supervisor-friendly dict.

    Preserves the fields that the Supervisor needs to make a verdict
    (role, textual content, tool call attempts and tool result markers)
    while dropping purely internal bookkeeping (``id``, ``expire_at``,
    ``source``) that would only add noise to the supervisor prompt.

    Args:
        message: The canonical message to serialize.

    Returns:
        A JSON-serializable dict with ``role`` and ``content`` keys plus,
        when applicable, ``tool_calls``, ``tool_call_id`` and
        ``tool_name`` keys.
    """
    serialized: dict[str, Any] = {
        "role": message.role,
        "content": message.content,
    }

    if message.tool_calls:
        tool_calls: list[dict[str, Any]] = []
        for tool_call in message.tool_calls:
            arguments_raw = tool_call.function.arguments
            arguments: Any
            if arguments_raw:
                try:
                    arguments = json.loads(arguments_raw)
                except json.JSONDecodeError:
                    arguments = arguments_raw
            else:
                arguments = {}
            tool_calls.append(
                {
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "arguments": arguments,
                }
            )
        serialized["tool_calls"] = tool_calls

    if message.tool_call_id is not None:
        serialized["tool_call_id"] = message.tool_call_id

    if message.tool_name is not None:
        serialized["tool_name"] = message.tool_name

    return serialized


def build_supervisor_context(
    system_prompt: str,
    messages: list[Message],
    tool_call: SupervisorToolCallAction | None = None,
    final_answer: str | None = None,
) -> str:
    """Serialize the full agent context for the Supervisor to evaluate.

    The supervisor principle (\"context 1:1 with what the model sees\")
    requires that the Supervisor receive the same enriched system prompt
    and full dialog history that the primary model observes, plus a
    structured description of the action under review. This function
    assembles that triple into a single JSON string suitable for passing
    as ``context=...`` to ``Provider.supervise()``.

    Exactly one of ``tool_call`` or ``final_answer`` must be provided:
    the former for tool-call supervision, the latter for final-answer
    supervision. Passing both, or neither, raises ``ValueError``.

    Args:
        system_prompt: The already-prepared system prompt that was (or
            will be) sent to the primary model. Callers MUST run the
            base system prompt through :func:`prepare_system_prompt`
            first; this function does not re-enrich it. Passing the raw
            template would silently violate the 1:1 context principle.
        messages: Canonical internal message history (the same
            ``list[Message]`` passed to the primary model), in
            chronological order.
        tool_call: The tool call the agent is about to make, or
            ``None`` when supervising a final answer.
        final_answer: The textual final answer the agent is about to
            return to the user, or ``None`` when supervising a tool
            call.

    Returns:
        A JSON string with keys ``system_prompt``, ``history`` (list of
        ``role``/``content``/optional ``tool_calls``/``tool_call_id``/
        ``tool_name`` dicts) and ``action`` (either
        ``{"type": "tool_call", "tool_name": ..., "args": ...}`` or
        ``{"type": "final_answer", "text": ...}``).

    Raises:
        ValueError: If both ``tool_call`` and ``final_answer`` are
            provided, or if neither is provided.
    """
    if (tool_call is None) == (final_answer is None):
        raise ValueError("build_supervisor_context requires exactly one of 'tool_call' or 'final_answer' to be set.")

    history: list[dict[str, Any]] = [_serialize_message_for_supervisor(message) for message in messages]

    if tool_call is not None:
        action: dict[str, Any] = {
            "type": "tool_call",
            "tool_name": tool_call.tool_name,
            "args": tool_call.args,
        }
    else:
        action = {
            "type": "final_answer",
            "text": final_answer,
        }

    payload: dict[str, Any] = {
        "system_prompt": system_prompt,
        "history": history,
        "action": action,
    }
    return json.dumps(payload, ensure_ascii=False)


async def send_llm_thoughts(thoughts: str, interface: UserInterface | None = None) -> None:
    if not gpt_settings.show_llm_thoughts:
        return None

    if not interface:
        return None

    if thoughts == "No content":
        return None

    message = f"💡💭 {thoughts}"

    await interface.send_message(message=message, reply=False)
    return None


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


def get_usage_from_responses_response(response_message: Response) -> UsageSchema:
    """Extract usage statistics from an OpenAI Responses API Response object.

    Args:
        response_message: The Response object returned by the OpenAI Responses API.

    Returns:
        A UsageSchema populated with token counts from the response.
    """
    if response_message.usage is None:
        return UsageSchema()

    usage = response_message.usage
    return UsageSchema(
        prompt_tokens=usage.input_tokens or 0,
        completion_tokens=usage.output_tokens or 0,
        total_tokens=usage.total_tokens or 0,
    )


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
