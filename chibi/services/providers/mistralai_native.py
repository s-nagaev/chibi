import base64
import binascii
import json
import random
from asyncio import sleep
from io import BytesIO
from typing import Any, Union, cast
from uuid import uuid4

import httpx
import httpx2
from loguru import logger
from mistralai.client import Mistral
from mistralai.client.errors import SDKError
from mistralai.client.models import (
    AssistantMessage,
    BaseModelCard,
    ChatCompletionChoice,
    ChatCompletionResponse,
    DocumentURLChunk,
    FTModelCard,
    FunctionCall,
    JSONSchemaTypedDict,
    ResponseFormatTypedDict,
    SystemMessage,
    TextChunk,
    Tool,
    ToolCall,
    ToolMessage,
    UsageInfo,
    UserMessage,
)
from openai.types.chat import ChatCompletionToolParam
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from chibi.config import application_settings, gpt_settings
from chibi.exceptions import (
    NoApiKeyProvidedError,
    NoResponseError,
    NotAuthorizedError,
    ServiceRateLimitError,
)
from chibi.interfaces.interface import UserInterface
from chibi.models import Message, User, mistral_content_to_text
from chibi.schemas.app import ChatResponseSchema, ModelChangeSchema, ModeratorsAnswer, VisionResultSchema
from chibi.services.metrics import MetricsService
from chibi.services.mistral_errors import classify_mistral_exception
from chibi.services.providers.provider import RestApiFriendlyProvider, ServiceResponseError
from chibi.services.providers.streaming import (
    AckSentinelStreamSession,
    delta_latched,
    delta_streaming_allowed,
    mark_non_streaming_fallback_spent,
    non_streaming_fallback_spent_error,
)
from chibi.services.providers.terminal_calls import (
    TERMINAL_TOOL_NAMES,
    append_repair_nudge,
    ensure_function_parameters,
    evaluate_turn_end,
    fold_terminal_calls_into_thoughts,
    forced_choice_applicable,
    honored_terminal_name,
    log_forced_choice_outcome,
    log_turn_misuse,
    offered_tool_definitions,
    partition_terminal_calls,
    plan_terminal_turn,
    resolve_turn_end_misuse,
)
from chibi.services.providers.tools import RegisteredChibiTools
from chibi.services.providers.tools.constants import MODERATOR_PROMPT
from chibi.services.providers.tools.schemas import ToolCallSchema
from chibi.services.providers.utils import (
    get_usage_from_mistral_response,
    get_usage_msg,
    prepare_system_prompt,
    send_llm_thoughts,
)
from chibi.services.usage_cache import UsageCacheStore

MistralMessageParam = Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage]

_TRANSCRIPTIONS_URL: str = "https://api.mistral.ai/v1/audio/transcriptions"
_TTS_URL: str = "https://api.mistral.ai/v1/audio/speech"


def _is_retryable_mistral_error(exception: BaseException) -> bool:
    """Return True for transient errors worth retrying: connection issues and HTTP 429/5xx.

    The Mistral SDK (3.3.0) exposes no per-status exception classes: any failed
    HTTP response surfaces as ``SDKError`` (a ``MistralError`` subclass) carrying
    the raw ``httpx2.Response``, so the status code is classified in
    ``chibi.services.mistral_errors`` instead. The client's own transport
    failures are plain ``httpx2`` exceptions (the SDK's HTTP library), which the
    project ``httpx`` classes above do not cover. HTTP 401/403 classify to the
    ``auth`` trigger and are deliberately NOT retried (owner decision): they
    fail over unconditionally on the first response.
    """
    if isinstance(exception, (ConnectionError, httpx.TransportError, httpx2.TransportError)):
        return True
    if isinstance(exception, SDKError):
        kind = classify_mistral_exception(exception)
        return kind in ("rate_limit", "server_error")
    return False


class MistralAI(RestApiFriendlyProvider):
    api_key = gpt_settings.mistralai_key
    chat_ready = True
    moderation_ready = True
    vision_ready = True
    ocr_ready = True
    stt_ready = True
    tts_ready = True

    name = "MistralAI"
    model_name_keywords = ["mistral", "mixtral", "ministral"]
    model_name_keywords_exclude = ["embed", "moderation", "ocr", "transcribe", "tts", "voxtral"]
    default_model = "mistral-small-2603"
    default_moderation_model = "mistral-moderation-2603"
    default_vision_model = "mistral-small-2603"
    default_ocr_model = "mistral-ocr-latest"
    default_stt_model = "voxtral-mini-transcribe-3-0"
    default_tts_model = "voxtral-mini-tts-3-0"
    default_tts_voice = "gb_jane_neutral"
    frequency_penalty: float | None = 0.6
    max_tokens: int = gpt_settings.max_tokens
    presence_penalty: float | None = 0.3
    temperature: float = 0.3

    def __init__(self, token: str) -> None:
        self._client: Mistral | None = None
        super().__init__(token=token)

    @property
    def tools_list(self) -> list[ChatCompletionToolParam]:
        """Return tools in OpenAI-compatible format (which Mistral uses)."""
        return RegisteredChibiTools.get_tool_definitions()

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    @property
    def client(self) -> Mistral:
        if self._client:
            return self._client

        if not self.token:
            raise NoApiKeyProvidedError(provider=self.name)

        self._client = Mistral(api_key=self.token)
        return self._client

    def _chat_request_kwargs(
        self,
        model: str,
        messages: list[MistralMessageParam],
        interface: UserInterface | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        """Build the request kwargs shared by every Mistral chat call.

        The streaming helper's non-streaming fallback re-issues the same
        request without the stream, so the streaming and non-streaming calls
        must be built by one builder: a divergence here would make the
        fallback answer a different request than the rejected stream.

        Args:
            model: The Mistral model identifier.
            messages: The conversation messages.
            interface: The active user interface, if any.
            max_tokens: Optional per-call output-token cap; overrides the
                instance-level value when provided, exact old behavior when None.

        Returns:
            Keyword arguments for ``chat.complete_async`` and ``chat.stream_async``.
        """
        offered = offered_tool_definitions(definitions=self.tools_list, interface=interface)
        forced_choice = forced_choice_applicable(interface=interface)
        if forced_choice:
            # Forced choice (optimisation only; the turn-end validator is the
            # guarantee): "any" demands a tool call without naming one, for the
            # same reason as the other adapters — the turn may end via either
            # terminal tool and may need work calls first. Gated on the single
            # switch AND interface presence (D1). The parameters field is
            # mandatory Mistral-SDK-side even when the schema is empty (recon C4),
            # so the forced path guarantees it on every offered tool.
            offered = ensure_function_parameters(definitions=offered)
        return {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "temperature": self.temperature,
            "tools": cast(list[Tool], offered),
            "tool_choice": "any" if forced_choice else "auto",
            "http_headers": {"Cache-Control": "max-age=86400"},
        }

    def get_thoughts(self, assistant_message: AssistantMessage) -> str | None:
        if not assistant_message.content:
            return None
        content = assistant_message.content
        if isinstance(content, str):
            return content

        if not isinstance(content, list):
            return None

        for chunk in content:
            if isinstance(chunk, TextChunk):
                return chunk.text
        return None

    async def _generate_content(
        self,
        model: str,
        messages: list[MistralMessageParam],
        interface: UserInterface | None = None,
        max_tokens: int | None = None,
    ) -> ChatCompletionResponse:
        """Generate Mistral content with empty-response retries.

        When live delta streaming is allowed for the request, the completion
        is requested via ``chat.stream_async`` and body-text deltas are
        emitted per chunk; the retry latch on the interface makes every
        attempt after the first emitted delta non-streaming.

        Args:
            model: The Mistral model identifier.
            messages: The conversation messages.
            interface: The active user interface, if any.
            max_tokens: Optional per-call output-token cap; overrides the
                instance-level value when provided, exact old behavior when None.

        Returns:
            The final aggregated Mistral response.

        Raises:
            NoResponseError: When every attempt returns an empty response.
        """
        for attempt in range(gpt_settings.retries):
            if delta_streaming_allowed(interface):
                response = await self._stream_generate_content(
                    model=model, messages=messages, interface=interface, max_tokens=max_tokens
                )
            else:
                response = await self.client.chat.complete_async(
                    **self._chat_request_kwargs(
                        model=model, messages=messages, interface=interface, max_tokens=max_tokens
                    )
                )

            if response.choices and len(response.choices) > 0:
                return response

            delay = gpt_settings.backoff_factor * (2**attempt)
            jitter = delay * random.uniform(0.1, 0.5)
            total_delay = delay + jitter

            logger.warning(
                f"Attempt #{attempt + 1}. Unexpected (empty) response received. Retrying in {total_delay} seconds..."
            )
            await sleep(total_delay)
        raise NoResponseError(provider=self.name, model=model, detail="Unexpected (empty) response received")

    async def _stream_generate_content(
        self,
        model: str,
        messages: list[MistralMessageParam],
        interface: UserInterface | None,
        max_tokens: int | None = None,
    ) -> ChatCompletionResponse:
        """Stream a Mistral completion and aggregate it into a final response.

        Emits one live delta per non-empty content chunk via the interface
        until the earliest tool-call signal in the stream (the full
        ``tool_calls`` typically arrive only on the final chunk). Delta
        content is normalized with ``mistral_content_to_text`` first, so a
        reasoning-capable model's list-form chunks (``ThinkChunk`` et al.)
        flatten to body text before the accumulator or the ACK gate sees
        them, and reasoning text never enters the answer. The chunks
        are rebuilt into a ChatCompletionResponse equivalent to the
        non-streaming response (fragmented tool-call arguments, parallel
        tool calls and usage included). The stream is always closed
        explicitly, also on cancellation.

        Args:
            model: The Mistral model identifier.
            messages: The conversation messages.
            interface: The active user interface, if any.
            max_tokens: Optional per-call output-token cap; overrides the
                instance-level value when provided, exact old behavior when None.

        Returns:
            The aggregated final ChatCompletionResponse.

        Raises:
            SDKError: When the streaming request is rejected after live deltas
                were already emitted (the retry latch forbids a transparent
                re-request at that point), when the rejection is neither a
                400 nor a 404, or when the single non-streaming fallback
                attempt of this provider has already failed (its error is
                re-raised on retry re-entry instead of issuing a second
                fallback; the latch is scoped to this provider, so other
                providers of the same failover walk keep their own).
        """
        content_parts: list[str] = []
        tool_call_slots: dict[int, dict[str, str]] = {}
        tool_call_signal = False
        finish_reason: str | None = None
        usage: UsageInfo | None = None
        completion_id = ""
        response_model = ""
        created = 0
        try:
            stream = await self.client.chat.stream_async(
                **self._chat_request_kwargs(model=model, messages=messages, interface=interface, max_tokens=max_tokens)
            )
        except SDKError as stream_error:
            # Some deployments reject the streaming request (e.g. an
            # unsupported streaming parameter) with a 400/404 before any
            # chunk can arrive. Mirroring the OpenAI path, the fallback to a
            # non-streaming request is handled here, inside the helper: the
            # delta latch decides whether a transparent re-request is still
            # allowed, and a failure after live deltas were emitted must
            # surface instead of silently re-issuing the request.
            status_code = stream_error.raw_response.status_code
            if delta_latched(interface) or status_code not in (400, 404):
                raise
            spent_error = non_streaming_fallback_spent_error(interface, self.name)
            if spent_error is not None:
                raise spent_error
            logger.warning(
                f"{self.name}/{model} rejected the streaming request ({stream_error}). "
                f"Falling back to a non-streaming request."
            )
            try:
                return await self.client.chat.complete_async(
                    **self._chat_request_kwargs(
                        model=model, messages=messages, interface=interface, max_tokens=max_tokens
                    )
                )
            except Exception as fallback_error:
                mark_non_streaming_fallback_spent(interface, self.name, fallback_error)
                raise
        async with AckSentinelStreamSession(interface) as deltas, stream:
            async for event in stream:
                chunk = event.data
                completion_id = chunk.id or completion_id
                response_model = chunk.model or response_model
                created = chunk.created or created
                if chunk.usage:
                    usage = chunk.usage
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                delta = choice.delta
                delta_text = mistral_content_to_text(delta.content)
                if delta_text:
                    content_parts.append(delta_text)
                    if not tool_call_signal:
                        await deltas.emit(delta_text)
                for tool_call_delta in delta.tool_calls or []:
                    tool_call_signal = True
                    index = tool_call_delta.index or 0
                    slot = tool_call_slots.setdefault(index, {"id": "", "name": "", "arguments": ""})
                    if tool_call_delta.id:
                        slot["id"] = tool_call_delta.id
                    if tool_call_delta.function:
                        if tool_call_delta.function.name:
                            slot["name"] = tool_call_delta.function.name
                        arguments = tool_call_delta.function.arguments
                        if isinstance(arguments, str):
                            slot["arguments"] += arguments
                        elif arguments:
                            slot["arguments"] = json.dumps(arguments)
        tool_calls = [
            ToolCall(
                id=slot["id"] or str(uuid4()),
                function=FunctionCall(
                    name=slot["name"],
                    arguments=slot["arguments"] or "{}",
                ),
            )
            for _, slot in sorted(tool_call_slots.items())
        ]
        message = AssistantMessage(content="".join(content_parts) or None, tool_calls=tool_calls or None)
        choice = ChatCompletionChoice(
            index=0,
            message=message,
            finish_reason=finish_reason or "stop",
        )
        return ChatCompletionResponse(
            id=completion_id or "streamed-completion",
            object="chat.completion",
            model=response_model,
            usage=usage,
            created=created or 0,
            choices=[choice],
        )

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=20, min=30, max=180),
        retry=retry_if_exception(_is_retryable_mistral_error),
        reraise=True,
    )
    async def get_chat_response(
        self,
        messages: list[Message],
        user: User,
        caller_storage_id: int,
        caller_thread_id: int,
        model: str | None = None,
        system_prompt: str = gpt_settings.assistant_prompt,
        interface: UserInterface | None = None,
        track_prompt_size: bool = False,
        max_tokens: int | None = None,
    ) -> tuple[ChatResponseSchema, list[Message]]:
        model = model or self.default_model

        history_size = len(messages)
        initial_messages = [msg.to_mistral() for msg in messages]
        chat_response, _ = await self._get_chat_completion_response(
            messages=initial_messages.copy(),
            user=user,
            model=model,
            system_prompt=system_prompt,
            interface=interface,
            conversation_messages=messages,
            track_prompt_size=track_prompt_size,
            caller_storage_id=caller_storage_id,
            caller_thread_id=caller_thread_id,
            max_tokens=max_tokens,
        )
        # Every loop iteration mirrors the messages it wants persisted into
        # the conversation list (including normalized terminal messages with
        # their visibility), so the list's delta is the persisted form of the
        # turn.
        new_messages = messages[history_size:]
        return (
            chat_response,
            [msg for msg in new_messages if msg.role != "system"],
        )

    async def _get_chat_completion_response(
        self,
        messages: list[MistralMessageParam],
        model: str,
        user: User,
        caller_storage_id: int,
        caller_thread_id: int,
        system_prompt: str = gpt_settings.assistant_prompt,
        interface: UserInterface | None = None,
        conversation_messages: list[Message] | None = None,
        track_prompt_size: bool = False,
        max_tokens: int | None = None,
        repair_used: bool = False,
    ) -> tuple[ChatResponseSchema, list[MistralMessageParam]]:
        """Run one model round: prepare the prompt, call the model and dispatch its tool calls.

        Args:
            messages: The conversation messages in Mistral wire format.
            model: The Mistral model identifier.
            user: The requesting user.
            caller_storage_id: Storage ID of the calling user.
            caller_thread_id: ID of the calling thread.
            system_prompt: The base system prompt.
            interface: The active user interface, if any.
            conversation_messages: The persisted conversation, if tool results must be folded back.
            track_prompt_size: Whether to record prompt usage.
            max_tokens: Optional per-call output-token cap.
            repair_used: Whether the turn-end repair round-trip has already been spent.

        Returns:
            The chat response schema and the updated wire message list.

        Raises:
            NoResponseError: When the model returns a choice without a message.
        """
        prepared_system_prompt = await prepare_system_prompt(
            base_system_prompt=system_prompt,
            user_id=user.id,
            interface=interface,
            conversation_messages=conversation_messages,
            thread_id=caller_thread_id,
        )
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=prepared_system_prompt, role="system")] + messages
        else:
            messages = [SystemMessage(content=prepared_system_prompt, role="system")] + messages[1:]

        response: ChatCompletionResponse = await self._generate_content(
            model=model, messages=messages, interface=interface, max_tokens=max_tokens
        )
        usage = get_usage_from_mistral_response(response_message=response)
        if track_prompt_size:
            UsageCacheStore().store(
                user_id=user.id,
                thread_id=interface.thread_id if interface else 0,
                usage=usage,
                provider=self.name,
            )

        if application_settings.is_influx_configured:
            MetricsService.send_usage_metrics(metric=usage, user=user, model=model, provider=self.name)

        message_data = response.choices[0].message
        if message_data is None:
            raise NoResponseError(provider=self.name, model=model, detail="Response choice carries no message")
        assistant_content = message_data.content or ""

        tool_calls = message_data.tool_calls
        if not tool_calls:
            if forced_choice_applicable(interface=interface):
                # A forced request answered with prose: the force was ignored.
                log_forced_choice_outcome(model=model, provider_name=self.name, interface=interface, outcome="ignored")
            # The validator judges the raw content, before the legacy
            # "no data" fallback: an empty ending must cost a repair
            # round-trip, never pass silently under a placeholder string.
            verdict = evaluate_turn_end(
                interface=interface,
                answer=assistant_content if isinstance(assistant_content, str) else "",
                ended_by_terminal_call=False,
                terminal_tool_name=None,
                conversation=conversation_messages or [],
            )
            if verdict.outcome == "misuse":
                return await resolve_turn_end_misuse(
                    verdict=verdict,
                    model=model,
                    provider_name=self.name,
                    interface=interface,
                    repair_used=repair_used,
                    append_nudge=lambda nudge_text: append_repair_nudge(
                        nudge_text=nudge_text,
                        messages=messages,
                        conversation_messages=conversation_messages,
                        to_wire=lambda text: UserMessage(content=text),
                    ),
                    reissue=lambda retried: self._get_chat_completion_response(
                        messages=messages,
                        model=model,
                        user=user,
                        system_prompt=system_prompt,
                        interface=interface,
                        conversation_messages=conversation_messages,
                        track_prompt_size=track_prompt_size,
                        caller_storage_id=caller_storage_id,
                        caller_thread_id=caller_thread_id,
                        max_tokens=max_tokens,
                        repair_used=retried,
                    ),
                )
            if verdict.misuse is not None:
                log_turn_misuse(
                    model=model,
                    provider_name=self.name,
                    interface=interface,
                    kind=verdict.misuse,
                    action="degraded_visible",
                )
            elif repair_used:
                log_turn_misuse(
                    model=model, provider_name=self.name, interface=interface, kind=None, action="recovered"
                )
            final_message = AssistantMessage(
                content=message_data.content or "",
            )
            messages.append(final_message)
            if conversation_messages is not None:
                conversation_messages.append(Message.from_mistral(final_message))
            return ChatResponseSchema(
                answer=message_data.content or "no data",
                provider=self.name,
                model=model,
                usage=usage,
            ), messages

        # Tool calls handling
        logger.log("CALL", f"{model} requested the call of {len(tool_calls)} tools.")

        thoughts = self.get_thoughts(assistant_message=message_data)
        if thoughts:
            await send_llm_thoughts(thoughts=thoughts, interface=interface)

        logger.log("THINK", f"{model}: {thoughts}. {get_usage_msg(usage=usage)}")

        calls = [
            ToolCallSchema(
                tool_name=tool_call.function.name,
                args=(
                    json.loads(tool_call.function.arguments)
                    if isinstance(tool_call.function.arguments, str)
                    else tool_call.function.arguments
                ),
            )
            for tool_call in tool_calls
        ]

        # Terminal tools (say / ack_silently) end the turn: they are never
        # dispatched and never receive a tool result — a result would send
        # the turn back to the model and reproduce the infinite ack loop
        # (design §3.2, recon C7). The branch sits between the parsed-calls
        # comprehension and the call_functions invocation on purpose.
        terminal_calls, work_calls = partition_terminal_calls(calls)
        if terminal_calls:
            if work_calls:
                # A backend ignored the parallel-tool-use ban and emitted a
                # terminal call alongside work calls: run the work calls,
                # drop the terminal entries and let the model terminate
                # cleanly on a later step (design §3.2, mixed-message case).
                logger.warning(
                    f"{model} emitted a terminal call ({terminal_calls[0].tool_name}) together with "
                    f"{len(work_calls)} work call(s); folding the terminal text and continuing the turn."
                )
                if isinstance(assistant_content, str):
                    assistant_content = fold_terminal_calls_into_thoughts(
                        thoughts=assistant_content, terminal_calls=terminal_calls
                    )
                tool_calls = [
                    tool_call for tool_call in tool_calls if tool_call.function.name not in TERMINAL_TOOL_NAMES
                ]
                calls = work_calls
            else:
                plan = plan_terminal_turn(terminal_calls=terminal_calls, thoughts=thoughts)
                logger.log("CALL", f"{model} called the terminal tool {terminal_calls[0].tool_name}; ending the turn.")
                if forced_choice_applicable(interface=interface):
                    log_forced_choice_outcome(
                        model=model, provider_name=self.name, interface=interface, outcome="honoured"
                    )
                verdict = evaluate_turn_end(
                    interface=interface,
                    answer=plan.answer,
                    ended_by_terminal_call=True,
                    terminal_tool_name=honored_terminal_name(terminal_calls),
                    conversation=conversation_messages or [],
                )
                if verdict.outcome != "misuse" and repair_used:
                    log_turn_misuse(
                        model=model, provider_name=self.name, interface=interface, kind=None, action="recovered"
                    )
                if conversation_messages is not None and plan.message is not None:
                    conversation_messages.append(plan.message)
                if verdict.outcome == "misuse":
                    return await resolve_turn_end_misuse(
                        verdict=verdict,
                        model=model,
                        provider_name=self.name,
                        interface=interface,
                        repair_used=repair_used,
                        append_nudge=lambda nudge_text: append_repair_nudge(
                            nudge_text=nudge_text,
                            messages=messages,
                            conversation_messages=conversation_messages,
                            to_wire=lambda text: UserMessage(content=text),
                        ),
                        reissue=lambda retried: self._get_chat_completion_response(
                            messages=messages,
                            model=model,
                            user=user,
                            system_prompt=system_prompt,
                            interface=interface,
                            conversation_messages=conversation_messages,
                            track_prompt_size=track_prompt_size,
                            caller_storage_id=caller_storage_id,
                            caller_thread_id=caller_thread_id,
                            max_tokens=max_tokens,
                            repair_used=retried,
                        ),
                    )
                return ChatResponseSchema(
                    answer=plan.answer, provider=self.name, model=model, usage=usage, visibility=plan.visibility
                ), messages

        results = await self.call_functions(
            calls=calls,
            caller_model=model,
            caller_provider=self.name,
            user_id=user.id,
            interface=interface,
            caller_storage_id=caller_storage_id,
            caller_thread_id=caller_thread_id,
        )

        for tool_call, result in zip(tool_calls, results):
            tool_call_message = AssistantMessage(
                content=assistant_content,
                tool_calls=[
                    ToolCall(
                        id=tool_call.id,
                        function=FunctionCall(
                            name=tool_call.function.name,
                            arguments=tool_call.function.arguments,
                        ),
                    )
                ],
            )
            tool_result_message = ToolMessage(
                name=tool_call.function.name,
                content=result.model_dump_json(),
                tool_call_id=tool_call.id,
            )
            messages.append(tool_call_message)
            messages.append(tool_result_message)
            if conversation_messages is not None:
                conversation_messages.append(Message.from_mistral(tool_call_message))
                conversation_messages.append(Message.from_mistral(tool_result_message))

        logger.log("CALL", "All the function results have been obtained. Returning them to the LLM...")
        return await self._get_chat_completion_response(
            messages=messages,
            model=model,
            user=user,
            system_prompt=system_prompt,
            interface=interface,
            conversation_messages=conversation_messages,
            track_prompt_size=track_prompt_size,
            caller_storage_id=caller_storage_id,
            caller_thread_id=caller_thread_id,
            max_tokens=max_tokens,
            repair_used=repair_used,
        )

    async def moderate_command(self, cmd: str, model: str | None = None) -> ModeratorsAnswer:
        moderator_model = model or self.default_moderation_model or self.default_model
        messages = [
            SystemMessage(content=MODERATOR_PROMPT, role="system"),
            Message(role="user", content=cmd).to_mistral(),
        ]
        response = await self.client.chat.complete_async(
            model=moderator_model,
            messages=messages,
            max_tokens=1024,
            temperature=0.2,
            response_format=ResponseFormatTypedDict(
                type="json_schema",
                json_schema=JSONSchemaTypedDict(
                    name="moderator_verdict",
                    schema_definition={
                        "type": "object",
                        "properties": {
                            "verdict": {"type": "string"},
                            "reason": {"type": "string"},
                            "status": {"type": "string", "default": "ok"},
                        },
                        "required": ["verdict"],
                    },
                    strict=True,
                ),
            ),
        )
        if not response.choices:
            return ModeratorsAnswer(status="error", verdict="declined", reason="no response from moderator received")

        usage = get_usage_from_mistral_response(response_message=response)

        if application_settings.is_influx_configured:
            MetricsService.send_usage_metrics(metric=usage, model=moderator_model, provider=self.name)

        message_data = response.choices[0].message
        answer = message_data.content if message_data else None
        if not answer:
            return ModeratorsAnswer(status="error", verdict="declined", reason="no response from moderator received")

        try:
            return ModeratorsAnswer.model_validate_json(answer, extra="ignore")  # type: ignore
        except Exception as e:
            msg = f"Error parsing moderator's response: {answer}. Error: {e}"
            logger.error(msg)
            return ModeratorsAnswer(verdict="declined", reason=msg, status="error")

    async def vision(
        self,
        image: bytes,
        mime_type: str,
        model: str | None = None,
        prompt: str | None = None,
    ) -> VisionResultSchema:
        model = model or self.default_vision_model
        prompt = prompt or "Describe the image in detail."
        logger.info(f"Analyzing image with model {model}...")

        # Encode image to base64
        image_base64 = base64.b64encode(image).decode("utf-8")
        data_url = f"data:{mime_type};base64,{image_base64}"

        # Use parse_async() for structured output with Pydantic models
        response = await self.client.chat.parse_async(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                }
            ],
            response_format=VisionResultSchema,
            max_tokens=16384,
        )

        if not response.choices:
            raise ServiceResponseError(
                provider=self.name,
                model=model,
                detail=f"Could not analyze image: empty response: {response}",
            )

        result = response.choices[0].message
        if not result or not result.parsed:
            raise ServiceResponseError(
                provider=self.name,
                model=model,
                detail=f"Could not analyze image: empty response: {response}",
            )

        logger.info(f"Image analyzed successfully: {result.parsed.short_description}\n{result.parsed.full_description}")
        return result.parsed

    async def ocr(self, pdf: bytes, model: str | None = None) -> VisionResultSchema:
        """Extract text from a PDF document using Mistral's OCR API.

        Args:
            pdf: The PDF file content as bytes.
            model: The model to use for OCR. Defaults to mistral-ocr-latest.

        Returns:
            VisionResultSchema containing the extracted text and descriptions.
        """
        model = model or self.default_ocr_model
        logger.info(f"[{self.name}] Extracting text from PDF with model {model}...")

        # Encode PDF to base64
        pdf_base64 = base64.b64encode(pdf).decode("utf-8")
        document_url = f"data:application/pdf;base64,{pdf_base64}"

        # Build the document using DocumentURLChunk
        document = DocumentURLChunk(document_url=document_url)

        try:
            # Use the OCR API to process the PDF
            response = await self.client.ocr.process_async(
                model=model,
                document=document,
            )

            if not response.pages:
                raise ServiceResponseError(
                    provider=self.name,
                    model=model,
                    detail="Could not extract text from PDF: empty response",
                )

            # Combine all pages' markdown content
            full_text = "\n\n".join(page.markdown for page in response.pages)

            # Create a short description from the first 100 characters
            short_description = full_text[:100].replace("\n", " ").strip()
            if len(full_text) > 100:
                short_description += "..."

            logger.info(f"[{self.name}] PDF text extracted successfully: {short_description}...")
            return VisionResultSchema(
                short_description=short_description,
                full_description=full_text[:1000] if len(full_text) > 1000 else full_text,
                text=full_text,
            )

        except ServiceResponseError:
            raise
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise ServiceResponseError(
                provider=self.name,
                model=model,
                detail=f"Could not extract text from PDF: {e}",
            )

    async def transcribe(self, audio: BytesIO, model: str | None = None) -> str:
        """Transcribe an audio recording with Mistral's Voxtral STT API.

        The Mistral SDK (3.3.0) exposes no audio namespace, so the call goes
        through raw ``httpx``: a multipart ``POST /v1/audio/transcriptions``
        with the model name as a form field and the recording as the ``file``
        field (the API requires exactly one of ``file|file_url|file_id``).

        Args:
            audio: The audio recording as a byte stream.
            model: The transcription model to use; falls back to the
                provider default when None.

        Returns:
            The transcribed text.

        Raises:
            NoApiKeyProvidedError: When no Mistral API key is configured.
            NotAuthorizedError: When the API rejects the key (HTTP 401).
            ServiceRateLimitError: When the API rate-limits the request (HTTP 429).
            ServiceResponseError: On any other HTTP or transport failure.
            ValueError: When the response carries no transcribed text.
        """
        model = model or self.default_stt_model
        logger.info(f"[{self.name}] Transcribing audio with model {model}...")

        if not self.token:
            raise NoApiKeyProvidedError(provider=self.name)

        try:
            async with self.get_async_httpx_client() as client:
                response = await client.post(
                    _TRANSCRIPTIONS_URL,
                    data={"model": model or ""},
                    files={"file": ("voice.ogg", audio.getvalue())},
                    headers=self._headers,
                )
        except Exception as e:
            logger.error(f"An error occurred while calling the {self.name} API: {e}")
            raise ServiceResponseError(provider=self.name, model=model or "", detail=str(e))

        if response.status_code == 401:
            raise NotAuthorizedError(provider=self.name, model=model or "")
        if response.status_code == 429:
            raise ServiceRateLimitError(provider=self.name, model=model or "")
        if response.status_code != 200:
            logger.error(
                f"Unexpected response from {self.name} API. Status code: {response.status_code}. Data: {response.text}"
            )
            raise ServiceResponseError(
                provider=self.name,
                model=model or "",
                detail=f"status_code: {response.status_code}, body: {response.text[:200]}",
            )

        payload: dict[str, Any] = response.json()
        text = payload.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError("Could not transcribe audio message")

        logger.info(f"[{self.name}] Transcribed text: {text}")
        return text

    async def speech(self, text: str, voice: str | None = None, model: str | None = None) -> bytes:
        """Generate speech audio from text with Mistral's Voxtral TTS API.

        The Mistral SDK (3.3.0) exposes no usable audio namespace for this
        flow, so the call goes through raw ``httpx``: a JSON ``POST
        /v1/audio/speech`` carrying ``model``, ``input``, ``voice_id`` and
        ``response_format``. The response carries the rendered audio as a
        base64 string in ``audio_data``, which is decoded to raw bytes.

        Args:
            text: The text to convert to speech.
            voice: The voice preset slug to use; falls back to the
                provider default when None.
            model: The TTS model to use; falls back to the provider
                default when None.

        Returns:
            The generated audio bytes in the requested format (mp3 by default).

        Raises:
            NoApiKeyProvidedError: When no Mistral API key is configured.
            NotAuthorizedError: When the API rejects the key (HTTP 401).
            ServiceRateLimitError: When the API rate-limits the request (HTTP 429).
            ServiceResponseError: On any other HTTP or transport failure.
            ValueError: When the response carries no decodable audio data.
        """
        model = model or self.default_tts_model
        voice = voice or self.default_tts_voice
        logger.info(f"[{self.name}] Generating speech with model {model} and voice {voice}...")

        if not self.token:
            raise NoApiKeyProvidedError(provider=self.name)

        try:
            async with self.get_async_httpx_client() as client:
                response = await client.post(
                    _TTS_URL,
                    json={
                        "model": model,
                        "input": text,
                        "voice_id": voice,
                        "response_format": "mp3",
                    },
                    headers=self._headers,
                )
        except Exception as e:
            logger.error(f"An error occurred while calling the {self.name} API: {e}")
            raise ServiceResponseError(provider=self.name, model=model, detail=str(e))

        if response.status_code == 401:
            raise NotAuthorizedError(provider=self.name, model=model)
        if response.status_code == 429:
            raise ServiceRateLimitError(provider=self.name, model=model)
        if response.status_code != 200:
            logger.error(
                f"Unexpected response from {self.name} API. Status code: {response.status_code}. Data: {response.text}"
            )
            raise ServiceResponseError(
                provider=self.name,
                model=model,
                detail=f"status_code: {response.status_code}, body: {response.text[:200]}",
            )

        payload: dict[str, Any] = response.json()
        audio_data = payload.get("audio_data")
        if not isinstance(audio_data, str) or not audio_data:
            raise ValueError("Could not generate speech audio")

        try:
            return base64.b64decode(audio_data)
        except (binascii.Error, ValueError) as e:
            logger.error(f"Malformed base64 audio data from the {self.name} API: {e}")
            raise ValueError("Could not generate speech audio") from e

    async def get_available_models(
        self, image_generation: bool = False, video_generation: bool = False
    ) -> list[ModelChangeSchema]:
        if image_generation or video_generation:
            return []

        try:
            response = await self.client.models.list_async()
            mistral_models = response.data or []
        except Exception as e:
            logger.error(f"Failed to get available models for provider {self.name} due to exception: {e}")
            return []

        model_cards = [model for model in mistral_models if isinstance(model, (BaseModelCard, FTModelCard))]
        all_models = [
            ModelChangeSchema(
                provider=self.name,
                name=model_card.id,
                display_name=model_card.id.replace("-", " ").title(),
                image_generation=False,
            )
            for model_card in model_cards
            if not self._model_name_has_keywords_exclude(model_name=model_card.id)
        ]
        return self.filter_and_return_list_of_models(models=all_models, image_generation=image_generation)
