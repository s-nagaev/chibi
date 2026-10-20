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

from chibi.config import application_settings, gpt_settings
from chibi.constants import PERSISTENT_MEMORY_PROMPT, SCHEDULER_HINT_PROMPT, TOOL_RESULTS_SHARE_WARN_THRESHOLD
from chibi.models import Message
from chibi.schemas.app import ModelChangeSchema, UsageSchema
from chibi.schemas.suno import SunoGetGenerationDetailsSchema
from chibi.services.context_measurement import estimate_messages_tokens
from chibi.services.context_note import format_context_note
from chibi.services.interface import UserInterface
from chibi.services.usage_cache import UsageCacheStore
from chibi.services.user import _is_tool_response_message, get_chibi_user
from chibi.storage.files import get_file_storage
from chibi.storage.files.file_storage import FileStorage
from chibi.utils.app import convert_list_of_models_to_str, get_available_skills

T = TypeVar("T")
P = ParamSpec("P")
M = TypeVar("M", bound=Callable[..., Coroutine[Any, Any, Any]])
AsyncFunc: TypeAlias = Callable[P, Coroutine[Any, Any, T]]


CONTEXT_OVERFLOW_STATUS_CODES: tuple[int, ...] = (400, 413)

# Markers restricted to token-count / context-length wordings. Generic
# phrases such as "request too large" or a bare "input token count" are
# deliberately absent: a non-overflow 400/413 whose body happens to contain
# them must keep its original error type instead of being reclassified.
CONTEXT_OVERFLOW_MARKERS: tuple[str, ...] = (
    "context_length_exceeded",
    "context length exceeded",
    "context length is exceeded",
    "maximum context length",
    "prompt is too long",
    "prompt too long",
    "exceeds the maximum number of tokens",
    "too many tokens",
    "reached max total tokens",
    "reduce the length of the messages",
)


def is_context_overflow_signal(status_code: int, detail: Any = None) -> bool:
    """Decide whether a provider error payload describes a context-length overflow.

    One shared mapping for every provider path: an overflow surfaces as an HTTP
    400 (or 413) whose body carries a context-length style error code or
    message, and providers word this differently. When the error cannot be
    recognized, the caller keeps its previous behaviour (generic provider
    error) instead of inventing a fragile heuristic.

    Args:
        status_code: HTTP status code carried by the error. Only 400 and 413
            can carry an overflow; any other code (or an unusable value) fails
            closed and the payload is never classified as an overflow.
        detail: Error payload: the SDK message, the parsed response body or
            any object whose string form contains the provider wording.
            ``None`` never matches.

    Returns:
        True when the status code plus message markers identify a context
        overflow.
    """
    if status_code not in CONTEXT_OVERFLOW_STATUS_CODES:
        return False
    if detail is None:
        return False
    text = str(detail).lower()
    return any(marker in text for marker in CONTEXT_OVERFLOW_MARKERS)


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


def _scheduler_hint_applicable() -> bool:
    """Check whether the scheduler hint belongs into the system prompt.

    Delegates to the same predicate that registers the scheduler agent tools,
    so the hint can never drift away from the actual tool availability: it is
    shown only when the scheduler tool gate is enabled AND the active client
    runner keeps the tools registered (the terminal REPL never does).

    Returns:
        True when the scheduler tools are registered in this process.
    """
    # Circular import avoidance: the tools package imports this module transitively via ChibiTool.
    from chibi.services.providers.tools.scheduler import _scheduler_tools_register

    return _scheduler_tools_register()


def _estimate_history_size(messages: list[Message]) -> int:
    """Estimate the total token size of a conversation history.

    A thin delegate to the single token-counting implementation
    (:func:`chibi.services.context_measurement.estimate_messages_tokens`),
    which includes ``tool_calls`` argument payloads that
    ``Message.estimate_tokens`` ignores (never persisted back to the model).

    Args:
        messages: The conversation messages.

    Returns:
        Estimated token count of the whole history, including assistant
        ``tool_calls`` arguments.
    """
    return estimate_messages_tokens(messages)


def _top_tool_result_offenders(messages: list[Message], limit: int = 3) -> list[tuple[str, int]]:
    """Collect the largest background tool-response turns (shape-(ii) blobs).

    Only tool name and estimated token count are returned — never the result
    content, to avoid leaking data into every system prompt.

    Args:
        messages: The conversation messages.
        limit: Maximum number of offenders to return.

    Returns:
        ``(tool_name, estimated_tokens)`` tuples, largest first.
    """
    offenders: list[tuple[str, int]] = []
    for msg in messages:
        if not _is_tool_response_message(msg):
            continue
        try:
            payload = json.loads(msg.content)
        except (TypeError, ValueError):
            continue
        tool_name = str(payload.get("tool_name") or "unknown")
        offenders.append((tool_name, msg.estimate_tokens))
    offenders.sort(key=lambda item: item[1], reverse=True)
    return offenders[:limit]


def collect_volatile_context_note(
    user_id: int,
    interface: UserInterface | None,
    conversation_messages: list[Message] | None = None,
) -> str:
    """Build the volatile per-turn context note.

    The note carries the keys that change on every turn: the approximate
    context size, the context-size warning (when the configured threshold is
    crossed) and the volume-based tool-results warning. It used to
    travel as system-payload keys, which invalidated prompt caching for the
    whole prefix on every turn; the provider entry points now append it to the
    last converted user message instead (see
    :func:`chibi.services.context_note.attach_context_note`).

    The warnings are informational: they report how full the context is and
    name the compaction tools. Compaction policy and timing live in the base
    system prompt, not here; the volatile note stays a statement of state.

    Args:
        user_id: The user identifier used to key the real context-size cache.
        interface: The user interface for the current request, or None.
            Sub-agent and other interface-less requests never carry the note,
            exactly as they never carried the volatile system keys.
        conversation_messages: Retained history used only as a fallback
            denominator for the volume-based tool-results warning when no
            real provider-reported usage is cached.

    Returns:
        The delimited note text, or an empty string when there is nothing to
        attach (no interface).
    """
    if interface is None:
        return ""
    thread_id = interface.thread_id
    real_context_size = UsageCacheStore().get(user_id=user_id, thread_id=thread_id)
    max_history_tokens = gpt_settings.max_history_tokens
    # Denominator for the volume-based tool-results share: the real
    # provider-reported context size when available, otherwise the local
    # per-message estimate (including tool_calls arguments).
    context_size_denominator = (
        real_context_size
        if real_context_size is not None
        else _estimate_history_size(conversation_messages)
        if conversation_messages
        else 0
    )
    lines: list[str] = []
    if real_context_size is not None:
        context_percentage = round(real_context_size / max_history_tokens * 100) if max_history_tokens else 0
        lines.append(
            f"approximate_context_size: {real_context_size:,} tokens "
            f"({context_percentage}% of {max_history_tokens:,} limit)"
        )
        if context_percentage > gpt_settings.context_size_warning_threshold:
            lines.append(
                "context_size_warning: "
                f"the context size is more than {gpt_settings.context_size_warning_threshold}% of "
                f"the maximum allowed ({max_history_tokens}) tokens. Compaction tools available in this "
                f"conversation: 'summarize_history' (history compression) and 'clear_tool_call_history' "
                f"(drops stale tool results)."
            )
    else:
        lines.append("approximate_context_size: n/a")

    if conversation_messages and context_size_denominator > 0:
        tool_results_tokens = sum(
            msg.estimate_tokens for msg in conversation_messages if _is_tool_response_message(msg)
        )
        offenders = _top_tool_result_offenders(conversation_messages)
        tool_share = tool_results_tokens / context_size_denominator
        if offenders and tool_share > TOOL_RESULTS_SHARE_WARN_THRESHOLD:
            share_percentage = round(tool_share * 100)
            top_offenders = "; ".join(f"{name}: ~{tokens} tokens" for name, tokens in offenders)
            lines.append(
                "tool_results_warning: "
                f"Background tool results occupy ~{tool_results_tokens} tokens "
                f"({share_percentage}% of the ~{context_size_denominator}-token context). "
                f"Largest offenders (tool name: estimated tokens): {top_offenders}. "
                f"Stale tool results can be dropped with 'clear_tool_call_history'."
            )
    return format_context_note(lines)


async def prepare_system_prompt(
    base_system_prompt: str,
    user_id: int,
    interface: UserInterface | None,
    conversation_messages: list[Message] | None = None,
    thread_id: int | None = None,
) -> str:
    """Prepare the system prompt payload sent to the LLM.

    The payload is stable across turns: the volatile keys (approximate context
    size, context-size warning, tool-results warning) do NOT travel here
    anymore. They are assembled per turn by
    :func:`collect_volatile_context_note` and attached by the provider entry
    points to the last converted user message, so the cached system prefix
    stays cacheable.

    Args:
        base_system_prompt: The base system prompt text.
        user_id: The user identifier used to fetch user metadata.
        interface: The user interface for the current request, or None.
        conversation_messages: Unused since the volatile keys moved out of the
            system payload; kept for signature compatibility with the provider
            call sites.
        thread_id: Session thread ID used when the interface is absent (e.g.
            sub-agent requests) so the effective working directory resolves to
            the same thread-scoped value as the parent request.

    Returns:
        JSON-encoded system prompt payload. The current thread's notes, when
        non-empty, are injected as the LAST payload key (the tail of the
        serialized JSON) to minimize prompt-cache disturbance; an empty notes
        string is treated the same as an absent one and the key is omitted.
    """
    user = await get_chibi_user(user_id=user_id)
    session_thread_id = interface.thread_id if interface else thread_id
    prompt: dict[str, Any] = {
        "system_prompt": base_system_prompt,
        "available_skills": get_available_skills(),
        "client": application_settings.client,
    }

    if application_settings.is_chroma_configured:
        prompt["system_prompt"] += PERSISTENT_MEMORY_PROMPT

    if _scheduler_hint_applicable():
        prompt["system_prompt"] += SCHEDULER_HINT_PROMPT

    if gpt_settings.filesystem_access:
        system_data = {
            "current_working_dir": user.get_effective_working_dir(session_thread_id),
            "platform": platform.platform(),
            "shell": os.environ.get("SHELL", "unknown"),
            "running_inside_container": application_settings.running_in_container,
        }
        if application_settings.running_in_container:
            system_data["container_type"] = application_settings.runtime_environment

        prompt["system"] = system_data

    if interface and getattr(interface, "uses_uploaded_file_storage", True):
        storage: FileStorage = get_file_storage(interface=interface)
        prompt["last_uploaded_files"] = await storage.get_available_files(limit=10)

    llms_data: list[ModelChangeSchema] = await user.get_available_models()
    prompt["available_models_to_delegate"] = convert_list_of_models_to_str(models=llms_data)

    prompt.update({"user_id": user.id, "user_info": user.info, "activated_skills": user.llm_skills})
    if session_thread_id is not None and (thread_notes := user.thread_notes.get(session_thread_id)):
        prompt["thread_notes"] = thread_notes
    return json.dumps(prompt)


async def send_llm_thoughts(thoughts: str, interface: UserInterface | None = None) -> None:
    """Forward LLM reasoning to the interface through the shared capture seam.

    The global ``show_llm_thoughts`` toggle governs interfaces that display
    thoughts as chat content (Telegram). Interfaces that capture thoughts for
    their own protocol (``captures_llm_thoughts``) always receive them and
    decide themselves what to do with the text.

    Args:
        thoughts: The LLM reasoning text to forward.
        interface: The active user interface, if any.
    """
    if not interface:
        return None

    if thoughts == "No content":
        return None

    if not gpt_settings.show_llm_thoughts and not interface.captures_llm_thoughts:
        return None

    await interface.send_llm_thoughts(thoughts)
    return None


def get_usage_from_anthropic_response(response_message: AnthropicMessage) -> UsageSchema:
    output_tokens = response_message.usage.output_tokens
    input_tokens = response_message.usage.input_tokens
    cache_creation_input_tokens = getattr(response_message.usage, "cache_creation_input_tokens", None) or 0
    cache_read_input_tokens = getattr(response_message.usage, "cache_read_input_tokens", None) or 0
    return UsageSchema(
        completion_tokens=output_tokens,
        prompt_tokens=input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        total_tokens=output_tokens + input_tokens + cache_creation_input_tokens + cache_read_input_tokens,
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
