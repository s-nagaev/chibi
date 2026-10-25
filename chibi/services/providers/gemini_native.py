import asyncio
import io
import math
import random
import wave
from asyncio import sleep
from collections.abc import AsyncGenerator
from contextlib import aclosing
from copy import copy
from io import BytesIO
from typing import Any, cast
from uuid import uuid4

import aiohttp
import httpx
from google.genai.client import AsyncClient, Client
from google.genai.errors import APIError
from google.genai.errors import ServerError as GeminiServerError
from google.genai.types import (
    Candidate,
    Content,
    ContentDict,
    ContentListUnion,
    ContentListUnionDict,
    FunctionCallDict,
    FunctionCallingConfig,
    FunctionDeclaration,
    FunctionResponseDict,
    GenerateContentConfig,
    GenerateContentResponse,
    GenerateContentResponseUsageMetadata,
    GenerateImagesConfig,
    GenerateImagesResponse,
    HttpOptions,
    Image,
    ImageConfig,
    Part,
    PartDict,
    PrebuiltVoiceConfig,
    Schema,
    SpeechConfig,
    Tool,
    ToolConfig,
    VoiceConfig,
)
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from chibi.config import application_settings, gpt_settings
from chibi.exceptions import NoResponseError, NotAuthorizedError, ServiceRateLimitError, ServiceResponseError
from chibi.models import Message, User
from chibi.schemas.app import ChatResponseSchema, ModelChangeSchema, ModeratorsAnswer, VisionResultSchema
from chibi.services.interface import UserInterface
from chibi.services.metrics import MetricsService
from chibi.services.providers.provider import RestApiFriendlyProvider
from chibi.services.providers.streaming import (
    AckSentinelStreamSession,
    delta_streaming_allowed,
)
from chibi.services.providers.terminal_calls import (
    TERMINAL_TOOL_NAMES,
    append_repair_nudge,
    evaluate_turn_end,
    fold_terminal_calls_into_thoughts,
    forced_choice_applicable,
    honored_terminal_name,
    log_forced_choice_outcome,
    log_turn_misuse,
    offered_tools,
    partition_terminal_calls,
    plan_terminal_turn,
    resolve_turn_end_misuse,
)
from chibi.services.providers.tools import RegisteredChibiTools
from chibi.services.providers.tools.constants import MODERATOR_PROMPT
from chibi.services.providers.tools.schemas import ToolCallSchema
from chibi.services.providers.utils import (
    get_usage_from_google_response,
    get_usage_msg,
    prepare_system_prompt,
    send_llm_thoughts,
)
from chibi.services.usage_cache import UsageCacheStore


def ensure_declaration_parameters(tools: list[Tool]) -> list[Tool]:
    """Guarantee every function declaration carries an explicit parameters object.

    Defensive, per recon C4: the "silent no-call when ``parameters`` is
    omitted" observation did NOT reproduce on the live re-run (downgraded to
    NOT REPRODUCED/UNVERIFIED), so this is verified-harmless defensiveness,
    not a measured failure mode. Every forced declaration always sends
    ``parameters={"type": "OBJECT", "properties": {}}``. The ``Tool`` objects
    come fresh from the ``tools_list`` property on every access, so mutating
    them here never touches shared state.

    Args:
        tools: Gemini Tool objects, each holding function declarations.

    Returns:
        The same list, with every declaration's parameters guaranteed set.
    """
    for tool in tools:
        for declaration in tool.function_declarations or []:
            if declaration.parameters is None:
                declaration.parameters = Schema(type="OBJECT", properties={})
    return tools


def _gemini_tool_name(tool: Tool) -> str:
    """Return the function name of a single-declaration Gemini tool.

    Args:
        tool: The Gemini Tool object (one declaration per tool, as built by
            ``tools_list``).

    Returns:
        The declaration's function name.
    """
    declarations = tool.function_declarations or []
    return str(declarations[0].name)


class Gemini(RestApiFriendlyProvider):
    api_key = gpt_settings.gemini_key
    chat_ready = True
    tts_ready = True
    stt_ready = True
    image_generation_ready = True
    moderation_ready = True
    vision_ready = True
    ocr_ready = True
    image_to_image_ready = True

    name = "Gemini"
    model_name_keywords = ["gemini", "gemma"]
    model_name_keywords_exclude = [
        "robotics",
        "computer",
        "audio",
        "stt",
        "image",
        "vision",
        "tts",
        "embedding",
        "2.0",
        "1.5",
    ]

    default_model = "models/gemini-3.8-flash"
    default_image_model = "models/imagen-4.0-fast-generate-001"
    default_tts_voice = "Kore"
    default_tts_model = "models/gemini-3.5-transcribe"
    default_stt_model = "models/gemini-3.8-flash"
    default_moderation_model = "models/gemini-3.5-flash-lite"
    default_vision_model = "models/gemini-3.8-flash"
    default_ocr_model = "models/gemini-3.8-flash"

    frequency_penalty: float | None = gpt_settings.frequency_penalty
    max_tokens: int = gpt_settings.max_tokens
    presence_penalty: float | None = gpt_settings.presence_penalty
    temperature: float = gpt_settings.temperature

    def __init__(self, token: str) -> None:
        super().__init__(token=token)

    @property
    def client_http_options(self) -> HttpOptions:
        """Build client-level HTTP options that force the httpx async backend.

        The google-genai SDK switches its async transport to aiohttp whenever aiohttp is
        importable and no client-level ``httpx_async_client`` is supplied. Config-level
        ``HttpOptions`` (e.g. inside ``GenerateContentConfig``) is consulted only for
        per-request retry settings and never selects the transport, so these options MUST
        be passed to every ``Client(...)`` constructor. A fresh ``httpx.AsyncClient`` is
        built on every access: an httpx client is stateful and cannot be reused after its
        context manager has closed it.

        Returns:
            Fresh ``HttpOptions`` carrying a new MITM-aware ``httpx.AsyncClient``.
        """
        return HttpOptions(httpx_async_client=self.get_async_httpx_client())

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
        self,
        model: str,
        contents: ContentListUnion | ContentListUnionDict,
        config: GenerateContentConfig,
        retry_server_errors: bool = False,
        interface: UserInterface | None = None,
    ) -> GenerateContentResponse:
        """Generate Gemini content with empty-response retries.

        When live delta streaming is allowed for the request, the completion
        is requested as a stream (inside the same client context) and
        body-text deltas are emitted per chunk; thought parts are never
        streamed. The retry latch on the interface makes every attempt after
        the first emitted delta non-streaming.

        Args:
            model: The Gemini model identifier.
            contents: The conversation contents.
            config: The generation config.
            retry_server_errors: When True, transient 5xx API errors are
                re-raised raw so the tenacity decorator on the chat path can
                retry them.
            interface: The active user interface, if any.

        Returns:
            The final aggregated Gemini response.

        Raises:
            NotAuthorizedError: When the API rejects the credentials.
            ServiceRateLimitError: When the API reports a per-day quota
                exhaustion.
            ServiceResponseError: When the API reports another error.
            NoResponseError: When every attempt returns an empty response.
        """
        for attempt in range(gpt_settings.retries):
            try:
                async with Client(api_key=gpt_settings.gemini_key, http_options=self.client_http_options).aio as client:
                    if delta_streaming_allowed(interface):
                        response: GenerateContentResponse = await self._generate_content_streamed(
                            client=client,
                            model=model,
                            contents=contents,
                            config=config,
                            interface=interface,
                        )
                    else:
                        response = await client.models.generate_content(
                            model=model,
                            contents=contents,
                            config=config,
                        )
                answer = self._get_text(response)
                if answer is not None or response.function_calls:
                    return response
                if answer is None and response.model_version == self.default_tts_model:
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

                elif retry_server_errors and isinstance(err.code, int) and err.code >= 500:
                    # Transient server error (5xx): re-raise the raw SDK error so the tenacity
                    # decorator on ``get_chat_response`` can retry it with exponential backoff.
                    # Scoped to the chat path (retry_server_errors=True): undecorated callers
                    # (moderation, speech, STT, vision, OCR) keep converting 5xx into
                    # ServiceResponseError, so users still get the specific provider-error
                    # message via ``handle_gpt_exceptions`` instead of the raw SDK exception.
                    raise

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

    async def _generate_content_streamed(
        self,
        client: AsyncClient,
        model: str,
        contents: ContentListUnion | ContentListUnionDict,
        config: GenerateContentConfig,
        interface: UserInterface | None,
    ) -> GenerateContentResponse:
        """Stream a Gemini completion inside an open client and aggregate it.

        The ``Client`` context stays open across the whole stream iteration.
        Emits one live delta per non-empty non-thought text part until the
        earliest function-call signal in the stream; thought parts are
        filtered out and never streamed. The chunks are rebuilt into a
        single GenerateContentResponse equivalent to the non-streaming
        response (body text, parallel function calls, thought signature,
        usage metadata and model version included). The stream is always
        closed explicitly, also on cancellation.

        Args:
            client: The open async Gemini client.
            model: The Gemini model identifier.
            contents: The conversation contents.
            config: The generation config.
            interface: The active user interface, if any.

        Returns:
            The aggregated final GenerateContentResponse.

        Raises:
            APIError: When the API request fails mid-stream.
        """
        parts: list[Part] = []
        tool_call_signal = False
        thought_signature: bytes | None = None
        usage_metadata: GenerateContentResponseUsageMetadata | None = None
        model_version: str | None = None
        stream = await client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config,
        )
        async with (
            AckSentinelStreamSession(interface) as deltas,
            aclosing(cast(AsyncGenerator[GenerateContentResponse, None], stream)) as chunks,
        ):
            async for chunk in chunks:
                if chunk.usage_metadata:
                    usage_metadata = chunk.usage_metadata
                if chunk.model_version:
                    model_version = chunk.model_version
                chunk_parts = self._get_stream_parts(response=chunk)
                for part in chunk_parts:
                    if part.function_call:
                        tool_call_signal = True
                    if signature := part.thought_signature:
                        thought_signature = signature
                    if part.function_call is None and (part.text is None or part.thought):
                        continue
                    parts.append(part)
                    if part.text and not part.function_call and not tool_call_signal:
                        await deltas.emit(part.text)
        if thought_signature:
            for part in parts:
                if part.function_call and not part.thought_signature:
                    part.thought_signature = thought_signature
        return GenerateContentResponse(
            candidates=[Candidate(content=Content(role="model", parts=parts))],
            usage_metadata=usage_metadata,
            model_version=model_version,
        )

    def _get_stream_parts(self, response: GenerateContentResponse) -> list[Part]:
        """Extract content parts from a stream chunk.

        Args:
            response: A single Gemini stream chunk.

        Returns:
            The parts carried by the chunk's first candidate, or an empty
            list when the chunk carries no parts.
        """
        if not response.candidates or not response.candidates[0].content or not response.candidates[0].content.parts:
            return []
        return list(response.candidates[0].content.parts)

    async def _get_chat_completion_response(
        self,
        messages: list[ContentDict],
        user: User,
        caller_storage_id: int,
        caller_thread_id: int,
        model: str | None = None,
        system_prompt: str = gpt_settings.assistant_prompt,
        interface: UserInterface | None = None,
        conversation_messages: list[Message] | None = None,
        track_prompt_size: bool = False,
        max_tokens: int | None = None,
        repair_used: bool = False,
    ) -> tuple[ChatResponseSchema, list[ContentDict]]:
        model_name = model or self.default_model

        prepared_system_prompt = await prepare_system_prompt(
            base_system_prompt=system_prompt,
            user_id=user.id,
            interface=interface,
            conversation_messages=conversation_messages,
            thread_id=caller_thread_id,
        )

        if "flash" in model_name and self.temperature > 0.4:
            temperature = 0.4
        else:
            temperature = self.temperature

        offered = (
            offered_tools(
                tools=self.tools_list,
                interface=interface,
                name_of=_gemini_tool_name,
            )
            if "gemini" in model_name
            else None
        )
        tool_config: ToolConfig | None = None
        if forced_choice_applicable(interface=interface) and offered is not None:
            # Forced choice for the agent loop (optimisation only; the
            # turn-end validator is the guarantee). Mode ANY without
            # allowed_function_names is the Gemini analog of OpenAI's
            # "required": the turn may legitimately end via either terminal
            # tool and may still need ordinary work calls first, so the
            # allowed set must not be narrowed to the terminal tools.
            # recon C1 verified mode ANY live (with an allowed-names list;
            # the unrestricted variant follows the same documented mode).
            # Defensive (recon C4): every forced declaration always carries an
            # explicit parameters object — verified harmless live, and the
            # validator keeps treating a 200-OK-with-empty-parts as an
            # R1-class failure regardless.
            ensure_declaration_parameters(tools=offered)
            tool_config = ToolConfig(function_calling_config=FunctionCallingConfig(mode="ANY"))

        generation_config = GenerateContentConfig(
            system_instruction=prepared_system_prompt if "gemini" in model_name else None,
            temperature=temperature,
            max_output_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            tools=offered,
            tool_config=tool_config,
        )

        response: GenerateContentResponse = await self._generate_content(
            model=model_name,
            contents=cast(ContentListUnionDict, messages),
            config=generation_config,
            retry_server_errors=True,
            interface=interface,
        )
        answer = self._get_text(response)
        usage = get_usage_from_google_response(response_message=response)
        if track_prompt_size:
            UsageCacheStore().store(
                user_id=user.id,
                thread_id=interface.thread_id if interface else 0,
                usage=usage,
                provider=self.name,
            )
        if application_settings.is_influx_configured:
            MetricsService.send_usage_metrics(metric=usage, model=model_name, provider=self.name, user=user)
        usage_message = get_usage_msg(usage=usage)

        if not response.function_calls:
            if forced_choice_applicable(interface=interface):
                # A forced request answered with prose: the force was ignored.
                log_forced_choice_outcome(
                    model=model_name, provider_name=self.name, interface=interface, outcome="ignored"
                )
            final_message: ContentDict = ContentDict(
                role="model",
                parts=[
                    PartDict(
                        text=answer,
                    )
                ],
            )
            messages.append(final_message)
            if conversation_messages is not None:
                conversation_messages.append(Message.from_google(final_message))
            verdict = evaluate_turn_end(
                interface=interface,
                answer=answer or "",
                ended_by_terminal_call=False,
                terminal_tool_name=None,
                conversation=conversation_messages or [],
            )
            if verdict.outcome == "misuse":
                return await resolve_turn_end_misuse(
                    verdict=verdict,
                    model=model_name,
                    provider_name=self.name,
                    interface=interface,
                    repair_used=repair_used,
                    append_nudge=lambda nudge_text: append_repair_nudge(
                        nudge_text=nudge_text,
                        messages=messages,
                        conversation_messages=conversation_messages,
                        to_wire=lambda text: ContentDict(role="user", parts=[PartDict(text=text)]),
                    ),
                    reissue=lambda retried: self._get_chat_completion_response(
                        messages=messages,
                        model=model_name,
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
                    model=model_name,
                    provider_name=self.name,
                    interface=interface,
                    kind=verdict.misuse,
                    action="degraded_visible",
                )
            elif repair_used:
                log_turn_misuse(
                    model=model_name, provider_name=self.name, interface=interface, kind=None, action="recovered"
                )
            return ChatResponseSchema(answer=answer, provider=self.name, model=model_name, usage=usage), messages

        # Tool calls handling
        logger.log("CALL", f"{model_name} requested the call of {len(response.function_calls)} tools.")

        if answer:
            await send_llm_thoughts(thoughts=answer, interface=interface)
        logger.log("THINK", f"{model_name}: {answer or 'No thoughts...'}. {usage_message}")

        function_calls = list(response.function_calls)
        calls = [
            ToolCallSchema(
                tool_name=str(function_call.name),
                args=copy(function_call.args),
            )
            for function_call in function_calls
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
                    f"{model_name} emitted a terminal call ({terminal_calls[0].tool_name}) together with "
                    f"{len(work_calls)} work call(s); folding the terminal text and continuing the turn."
                )
                answer = fold_terminal_calls_into_thoughts(thoughts=answer, terminal_calls=terminal_calls)
                function_calls = [
                    function_call
                    for function_call in function_calls
                    if str(function_call.name) not in TERMINAL_TOOL_NAMES
                ]
                calls = work_calls
            else:
                plan = plan_terminal_turn(terminal_calls=terminal_calls, thoughts=answer)
                logger.log(
                    "CALL", f"{model_name} called the terminal tool {terminal_calls[0].tool_name}; ending the turn."
                )
                if forced_choice_applicable(interface=interface):
                    log_forced_choice_outcome(
                        model=model_name, provider_name=self.name, interface=interface, outcome="honoured"
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
                        model=model_name, provider_name=self.name, interface=interface, kind=None, action="recovered"
                    )
                if conversation_messages is not None and plan.message is not None:
                    conversation_messages.append(plan.message)
                if verdict.outcome == "misuse":
                    return await resolve_turn_end_misuse(
                        verdict=verdict,
                        model=model_name,
                        provider_name=self.name,
                        interface=interface,
                        repair_used=repair_used,
                        append_nudge=lambda nudge_text: append_repair_nudge(
                            nudge_text=nudge_text,
                            messages=messages,
                            conversation_messages=conversation_messages,
                            to_wire=lambda text: ContentDict(role="user", parts=[PartDict(text=text)]),
                        ),
                        reissue=lambda retried: self._get_chat_completion_response(
                            messages=messages,
                            model=model_name,
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
                return (
                    ChatResponseSchema(
                        answer=plan.answer,
                        provider=self.name,
                        model=model_name,
                        usage=usage,
                        visibility=plan.visibility,
                    ),
                    messages,
                )

        results = await self.call_functions(
            calls=calls,
            caller_model=model_name,
            caller_provider=self.name,
            user_id=user.id,
            interface=interface,
            caller_storage_id=caller_storage_id,
            caller_thread_id=caller_thread_id,
        )

        thought_signature = self._get_thought_signature(response=response)
        if not thought_signature and "gemini-3" in model_name:
            logger.warning(
                f"Could not get thought signature for function call, no response candidates found: "
                f"{response.candidates}."
            )

        for function_call, result in zip(function_calls, results):
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
            if conversation_messages is not None:
                conversation_messages.append(Message.from_google(tool_call_message))
                conversation_messages.append(Message.from_google(tool_result_message))

        logger.log("CALL", "All the function results have been obtained. Returning them to the LLM...")
        return await self._get_chat_completion_response(
            messages=messages,
            model=model_name,
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

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=20, min=30, max=180),
        retry=retry_if_exception_type(
            (ConnectionError, aiohttp.ServerDisconnectedError, httpx.TransportError, GeminiServerError)
        ),
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
        initial_messages = [msg.to_google() for msg in messages]

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
        return chat_response, new_messages

    async def _generate_image_via_content_creation_model(
        self,
        prompt: str,
        model: str,
        images: list[tuple[bytes, str]] | None = None,
    ) -> list[Image]:
        """Generate images with a Nano Banana content-creation model.

        Args:
            prompt: Text description of the image to generate.
            model: The Nano Banana content-creation model name.
            images: Optional reference images as (bytes, mime type) tuples,
                passed to the model as inline parts after the prompt.

        Returns:
            List of images generated by the model.

        Raises:
            ServiceResponseError: When the model response contains no content parts.
        """
        image_size = (
            gpt_settings.image_size_nano_banana if "flash" not in model else None
        )  # flash-models don't support it

        generation_config = GenerateContentConfig(
            image_config=ImageConfig(
                aspect_ratio=gpt_settings.image_aspect_ratio,
                image_size=image_size,
            )
        )
        image_parts = [Part.from_bytes(data=data, mime_type=mime_type) for data, mime_type in (images or [])]

        async with Client(api_key=gpt_settings.gemini_key, http_options=self.client_http_options).aio as client:
            response: GenerateContentResponse = await client.models.generate_content(
                model=model,
                contents=cast(ContentListUnion, [prompt, *image_parts]),
                config=generation_config,
            )
        if not response.parts:
            raise ServiceResponseError(provider=self.name, model=model, detail="No content-parts in response found")

        response_images: list[Image | None] = [part.as_image() for part in response.parts if part]
        return [image for image in response_images if image]

    async def _generate_image_by_imagen(
        self,
        prompt: str,
        model: str,
    ) -> list[Image]:
        if "preview" in model or "fast" in model:
            image_size = None
        else:
            image_size = gpt_settings.image_size_imagen

        generation_config = GenerateImagesConfig(
            aspect_ratio=gpt_settings.image_aspect_ratio,
            number_of_images=gpt_settings.image_n_choices,
            image_size=image_size,
        )
        async with Client(api_key=gpt_settings.gemini_key, http_options=self.client_http_options).aio as client:
            response: GenerateImagesResponse = await client.models.generate_images(
                model=model,
                prompt=prompt,
                config=generation_config,
            )
            images_in_response = response.images

            return [image for image in images_in_response if image]

    async def moderate_command(self, cmd: str, model: str | None = None) -> ModeratorsAnswer:
        moderator_model = model or self.default_moderation_model or self.default_model

        generation_config = GenerateContentConfig(
            system_instruction=MODERATOR_PROMPT,
            temperature=0.1,
            max_output_tokens=1024,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            response_schema=ModeratorsAnswer,
        )
        messages = [
            Message(role="user", content=cmd).to_google(),
        ]
        response: GenerateContentResponse = await self._generate_content(
            model=moderator_model,
            contents=cast(ContentListUnionDict, messages),
            config=generation_config,
        )
        answer = self._get_text(response)
        if not answer:
            return ModeratorsAnswer(verdict="declined", reason="no moderator answer received", status="error")

        answer = answer.strip("```").strip("json").strip()
        usage = get_usage_from_google_response(response_message=response)
        if application_settings.is_influx_configured:
            MetricsService.send_usage_metrics(metric=usage, model=moderator_model, provider=self.name)

        try:
            return ModeratorsAnswer.model_validate_json(answer)
        except Exception as e:
            logger.error(f"Error parsing moderator's response: {answer}. Error: {e}")
            return ModeratorsAnswer(verdict="declined", reason=answer, status="error")

    async def get_images(
        self, prompt: str, model: str | None = None, images: list[tuple[bytes, str]] | None = None
    ) -> list[BytesIO]:
        """Generate images from a text prompt and optional reference images.

        Args:
            prompt: Text description of the image to generate.
            model: Image model name. Defaults to the provider's default image model.
            images: Optional reference images as (bytes, mime type) tuples. Supported
                by Nano Banana content-creation models only.

        Returns:
            List of BytesIO objects with image data.

        Raises:
            ServiceResponseError: When reference images are provided for an
                ``imagen-*`` model, which does not accept image input.
        """
        selected_model = model or self.default_image_model

        if "imagen-" in selected_model:
            if images:
                raise ServiceResponseError(
                    provider=self.name,
                    model=selected_model,
                    detail="imagen models do not support image input; select a Nano Banana image model instead",
                )
            generated_images = await self._generate_image_by_imagen(prompt=prompt, model=selected_model)
        else:
            generated_images = await self._generate_image_via_content_creation_model(
                prompt=prompt, model=selected_model, images=images
            )

        return [BytesIO(image.image_bytes) for image in generated_images if image.image_bytes]

    @classmethod
    def is_image_ready_model(cls, model_name: str) -> bool:
        return "image" in model_name

    @classmethod
    def supports_image_input(cls, model: str) -> bool:
        """Check whether a specific model of this provider accepts image input for generation.

        Args:
            model: Model name to check.

        Returns:
            True for Nano Banana content-creation image models, False for
            ``imagen-*`` models and everything else.
        """
        name = model.removeprefix("models/")
        if name.startswith("imagen-"):
            return False
        return "image" in name

    def get_model_display_name(self, model_name: str) -> str:
        model_name_model_market_name_mapping = {
            "models/gemini-3-pro-image": "Nano Banana Pro",
            "models/gemini-3.1-flash-image": "Nano Banana 2",
            "models/gemini-2.5-flash-image": "Nano Banana",
        }
        if "gemini" in model_name.lower():
            for k, v in model_name_model_market_name_mapping.items():
                if k in model_name:
                    display_name = model_name.replace(k, v)
                    return super().get_model_display_name(model_name=display_name)

        if "imagen" in model_name:
            display_name = model_name.removeprefix("models/").rsplit("-generate-", 1)[0].replace("-", " ").title()
            return super().get_model_display_name(model_name=display_name)

        return super().get_model_display_name(model_name=model_name[7:])

    async def get_available_models(
        self, image_generation: bool = False, video_generation: bool = False
    ) -> list[ModelChangeSchema]:
        try:
            async with Client(api_key=gpt_settings.gemini_key, http_options=self.client_http_options).aio as aclient:
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
                video_generation=False,
            )
            async for model in models
            if model.name
        ]
        return self.filter_and_return_list_of_models(
            models=all_models, image_generation=image_generation, video_generation=video_generation
        )

    async def speech(self, text: str, voice: str | None = None, model: str | None = None) -> bytes:
        voice = voice or self.tts_voice
        model = model or self.tts_model
        logger.info(f"Recording a voice message with model {model}...")

        generation_config = GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=SpeechConfig(
                voice_config=VoiceConfig(
                    prebuilt_voice_config=PrebuiltVoiceConfig(
                        voice_name=voice,
                    )
                )
            ),
        )

        response: GenerateContentResponse = await self._generate_content(
            model=model,
            contents=f"TTS the following: {text}",
            config=generation_config,
        )

        if not response.parts:
            raise ServiceResponseError(
                provider=self.name,
                model=model,
                detail=f"Gemini API returned a response w/o parts data: {response.model_dump()}",
            )

        response_part: Part = response.parts[0]
        if not response_part.inline_data:
            raise ServiceResponseError(
                provider=self.name,
                model=model,
                detail=f"No inline data found in the response.parts[0]: {response_part}",
            )

        data: bytes | None = response_part.inline_data.data

        if not data:
            raise ServiceResponseError(
                provider=self.name,
                model=model,
                detail=f"No data found inside the Blob object (inline data): {response_part.inline_data}",
            )

        buf = io.BytesIO()

        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)  # PCM16 = 2 bytes
            w.setframerate(24000)
            w.writeframes(data)

        return buf.getvalue()

    async def transcribe(self, audio: BytesIO, model: str | None = None) -> str:
        model = model or self.default_stt_model
        logger.info(f"Transcribing audio with model {model}...")

        generation_config = GenerateContentConfig()

        response = await self._generate_content(
            model=model,
            contents=cast(
                ContentListUnion,
                [
                    "STT this audio clip",
                    Part.from_bytes(
                        data=audio.read(),
                        mime_type="audio/ogg",  # Telegram format
                    ),
                ],
            ),
            config=generation_config,
        )

        if application_settings.log_prompt_data:
            logger.info(f"Transcribed text: {response.text}")

        if result := response.text:
            return result

        raise ServiceResponseError(
            provider=self.name,
            model=model,
            detail=f"Could not transcribe audio message: empty text data in response: {response}",
        )

    async def vision(
        self, image: bytes, mime_type: str, model: str | None = None, prompt: str | None = None
    ) -> VisionResultSchema:
        model = model or self.default_vision_model
        prompt = prompt or "Describe the image in detail"
        logger.info(f"[{self.name}] Analyzing image with model {model}...")

        generation_config = GenerateContentConfig(
            response_schema=VisionResultSchema,
        )

        response = await self._generate_content(
            model=model,
            contents=cast(
                ContentListUnion,
                [
                    Part.from_bytes(
                        data=image,
                        mime_type=mime_type,
                    ),
                    prompt,
                ],
            ),
            config=generation_config,
        )

        result = response.text
        if not result:
            raise ServiceResponseError(
                provider=self.name,
                model=model,
                detail=f"Could not analyze image: empty text data in response: {response}",
            )

        try:
            parsed_result = VisionResultSchema.model_validate_json(json_data=result, extra="ignore")
        except Exception as e:
            raise ServiceResponseError(
                provider=self.name,
                model=model,
                detail=f"Could not analyze image or parse a result due to exception: {e}",
            ) from e

        logger.info(f"[{self.name}] Analyzed image: {parsed_result.short_description}")

        return parsed_result

    async def ocr(self, pdf: bytes, model: str | None = None) -> VisionResultSchema:
        """Extract text from a PDF document using Google Gemini's document vision.

        Args:
            pdf: The PDF file content as bytes.
            model: The model to use for OCR. Defaults to default_vision_model.

        Returns:
            VisionResultSchema containing the extracted text and descriptions.
        """
        model = model or self.default_ocr_model
        logger.info(f"[{self.name}] Extracting text from PDF with model {model}...")

        generation_config = GenerateContentConfig(
            response_schema=VisionResultSchema,
        )
        response = await self._generate_content(
            model=model,
            contents=cast(
                ContentListUnion,
                [
                    Part.from_bytes(
                        data=pdf,
                        mime_type="application/pdf",
                    ),
                    "Extract all text from this PDF document.",
                ],
            ),
            config=generation_config,
        )

        result = response.text
        if not result:
            raise ServiceResponseError(
                provider=self.name,
                model=model,
                detail="Could not extract text from PDF: empty text data in response",
            )

        try:
            parsed_result = VisionResultSchema.model_validate_json(json_data=result, extra="ignore")
        except Exception as e:
            raise ServiceResponseError(
                provider=self.name,
                model=model,
                detail=f"Could not extract text from PDF or parse a result due to exception: {e}",
            ) from e

        logger.info(f"[{self.name}] PDF text extracted successfully: {parsed_result.short_description}")

        return parsed_result
