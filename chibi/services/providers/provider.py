import asyncio
import base64
import inspect
import json
import random
import re
from abc import ABC
from asyncio import sleep
from functools import wraps
from io import BytesIO
from typing import Any, Awaitable, Callable, Generic, Literal, Optional, ParamSpec, TypeVar, cast
from urllib.parse import urljoin

import httpx
from anthropic import AsyncClient, NotGiven, Omit
from anthropic.types import (
    CacheControlEphemeralParam,
    MessageParam,
    TextBlock,
    TextBlockParam,
    ToolChoiceToolParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlock,
)
from anthropic.types import (
    Message as AnthropicMessage,
)
from anthropic.types.tool_param import InputSchemaTyped
from httpx import Response
from httpx._types import QueryParamTypes, RequestData
from loguru import logger
from openai import (
    APIConnectionError,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
    omit,
)
from openai import NotGiven as OpenAINotGiven
from openai import Omit as OpenAIOmit
from openai.types import Image, ImagesResponse, ReasoningEffort
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
)
from openai.types.chat.chat_completion import ChatCompletion, Choice
from pydantic import BaseModel

from chibi.config import application_settings, gpt_settings
from chibi.constants import IMAGE_SIZE_OPENAI_LITERAL
from chibi.exceptions import (
    NoApiKeyProvidedError,
    NoModelSelectedError,
    NoResponseError,
    NotAuthorizedError,
    ServiceConnectionError,
    ServiceRateLimitError,
    ServiceResponseError,
)
from chibi.models import FunctionSchema, Message, ToolSchema, User
from chibi.schemas.app import (
    ChatResponseSchema,
    ModelChangeSchema,
    ModeratorsAnswer,
    SupervisorAnswer,
    SupervisorVerdict,
    VisionResultSchema,
)
from chibi.services.interface import UserInterface
from chibi.services.metrics import MetricsService
from chibi.services.providers.tools import RegisteredChibiTools
from chibi.services.providers.tools.constants import MODERATOR_PROMPT, SUPERVISOR_PROMPT
from chibi.services.providers.tools.schemas import ToolCallSchema, ToolResponseSchema
from chibi.services.providers.utils import (
    SupervisorToolCallAction,
    build_supervisor_context,
    get_usage_from_anthropic_response,
    get_usage_from_openai_response,
    get_usage_msg,
    prepare_system_prompt,
    send_llm_thoughts,
)

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T", bound=BaseModel)


class RegisteredProviders:
    all: dict[str, type["Provider"]] = {}
    available: dict[str, type["Provider"]] = {}

    def __init__(self, user_api_keys: dict[str, str] | None = None) -> None:
        self.tokens = {} if not user_api_keys else user_api_keys
        if gpt_settings.public_mode:
            self.available: dict[str, type["Provider"]] = {
                provider.name.lower(): provider
                for provider in RegisteredProviders.all.values()
                if provider.name in self.tokens
            }

    @classmethod
    def register(cls, provider: type["Provider"]) -> None:
        cls.all[provider.name.lower()] = provider

    @classmethod
    def register_as_available(cls, provider: type["Provider"]) -> None:
        cls.available[provider.name.lower()] = provider

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
    def vision_ready(self) -> dict[str, type["Provider"]]:
        return {provider_name: provider for provider_name, provider in self.available.items() if provider.vision_ready}

    @property
    def image_generation_ready(self) -> dict[str, type["Provider"]]:
        return {name: provider for name, provider in self.available.items() if provider.image_generation_ready}

    @property
    def stt_ready(self) -> dict[str, type["Provider"]]:
        return {name: provider for name, provider in self.available.items() if provider.stt_ready}

    @property
    def tts_ready(self) -> dict[str, type["Provider"]]:
        return {name: provider for name, provider in self.available.items() if provider.tts_ready}

    @property
    def ocr_ready(self) -> dict[str, type["Provider"]]:
        return {name: provider for name, provider in self.available.items() if provider.ocr_ready}

    def get_instance(self, provider: type["Provider"]) -> Optional["Provider"]:
        api_key = self.get_api_key(provider)
        if not api_key:
            return None
        return provider(token=api_key)

    def get(self, provider_name: str) -> Optional["Provider"]:
        if provider_name.lower() not in self.available:
            return None
        provider = self.available[provider_name.lower()]
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

    @property
    def first_supervisor_ready(self) -> Optional["Provider"]:
        """Resolve the supervisor provider instance using the full fallback chain.

        Resolution order:

        1. Explicit ``supervisor_provider`` from config (or its fallback
           to ``moderation_provider`` via ``supervisor_provider_resolved``)
           -- instantiate that provider class.
        2. Fallback to ``first_moderation_ready`` (existing default picker
           among moderation-ready providers).

        This property complements the config-level resolution in
        ``GPTSettings`` by verifying that the resolved provider is actually
        available at runtime (has an API key, is registered, etc.).

        Returns:
            A ``Provider`` instance ready to call ``supervise()``, or
            ``None`` if no eligible provider could be resolved.
        """
        supervisor_provider_name = gpt_settings.supervisor_provider_resolved
        if supervisor_provider_name:
            provider_class = self.available.get(supervisor_provider_name.lower())
            if provider_class is not None:
                instance = self.get_instance(provider=provider_class)
                if instance is not None:
                    return instance
        return self.first_moderation_ready

    @property
    def first_vision_ready(self) -> Optional["Provider"]:
        if provider := next(iter(self.vision_ready.values()), None):
            return self.get_instance(provider=provider)
        return None

    @property
    def first_ocr_ready(self) -> Optional["Provider"]:
        if provider := next(iter(self.ocr_ready.values()), None):
            return self.get_instance(provider=provider)
        return None


class Provider(ABC):
    api_key: str | None = None
    stt_ready: bool = False
    tts_ready: bool = False
    ocr_ready: bool = False
    chat_ready: bool = False
    vision_ready: bool = False
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
    default_vision_model: str | None = None
    default_ocr_model: str | None = None

    timeout: int = gpt_settings.timeout

    def __init__(self, token: str) -> None:
        self.token = token

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if hasattr(cls, "name"):
            RegisteredProviders.register(cls)

        if cls.api_key:
            RegisteredProviders.register_as_available(cls)

    @property
    def stt_model(self) -> str:
        if model := (gpt_settings.stt_model or self.default_stt_model):
            return model
        raise ValueError("No default STT model set")

    @property
    def tts_model(self) -> str:
        if model := (gpt_settings.tts_model or self.default_tts_model):
            return model
        raise ValueError("No default TTS model set")

    @property
    def tts_voice(self) -> str:
        if voice := (gpt_settings.tts_voice or self.default_tts_voice):
            return voice
        raise ValueError("No default TTS voice set")

    async def _get_chat_response_impl(
        self,
        messages: list[Message],
        user: User,
        model: str | None = None,
        system_prompt: str = gpt_settings.assistant_prompt,
        interface: UserInterface | None = None,
    ) -> tuple[ChatResponseSchema, list[Message]]:
        raise NotImplementedError

    async def get_chat_response(
        self,
        messages: list[Message],
        user: User,
        model: str | None = None,
        system_prompt: str = gpt_settings.assistant_prompt,
        interface: UserInterface | None = None,
        supervisor_retry_count: int = 0,
    ) -> tuple[ChatResponseSchema, list[Message]]:
        """Get a chat response with optional Supervisor review of the final answer.

        This is the public entry point for chat completion. Concrete provider
        families implement :meth:`_get_chat_response_impl`; this method wraps
        the result with the final-answer Supervisor hook when
        ``gpt_settings.supervisor_enabled`` is True.

        When the Supervisor returns an ``intervene`` verdict, a synthetic
        feedback message is appended and generation is retried recursively up
        to ``gpt_settings.max_supervisor_retries`` times. Rejected drafts and
        synthetic feedback messages are part of the retry context but are NOT
        returned in ``new_messages``, so they are not persisted to storage.

        Args:
            messages: Conversation history in canonical format.
            user: The user requesting the response.
            model: Optional model override.
            system_prompt: Base system prompt template.
            interface: Optional interface for progress/thoughts.
            supervisor_retry_count: Internal retry counter for Supervisor
                interventions. Callers should leave this at the default.

        Returns:
            A tuple of the final chat response and the list of new messages
            produced by this call (suitable for persistence).

        Raises:
            NotImplementedError: If the provider family does not implement
                :meth:`_get_chat_response_impl` (e.g. Cloudflare).
        """
        response, new_messages = await self._get_chat_response_impl(
            messages=messages,
            user=user,
            model=model,
            system_prompt=system_prompt,
            interface=interface,
        )

        if not gpt_settings.supervisor_enabled:
            return response, new_messages

        supervisor = RegisteredProviders().first_supervisor_ready
        if supervisor is None:
            logger.warning("Supervisor is enabled but no supervisor provider could be resolved; running without it.")
            return response, new_messages

        prepared_system_prompt = ""
        if system_prompt:
            prepared_system_prompt = await prepare_system_prompt(
                base_system_prompt=system_prompt, user_id=user.id, interface=interface
            )

        full_history = list(messages) + list(new_messages)
        supervisor_history = full_history[:-1] if new_messages else full_history
        context = build_supervisor_context(
            system_prompt=prepared_system_prompt,
            messages=supervisor_history,
            final_answer=response.answer,
        )
        verdict = await supervisor.supervise(context=context)

        if verdict.verdict != SupervisorVerdict.INTERVENE:
            return response, new_messages

        reason = verdict.reason or "Supervisor intervened."
        logger.warning(f"Supervisor intervened on final answer: {reason}")

        if supervisor_retry_count >= gpt_settings.max_supervisor_retries:
            logger.warning(
                f"Supervisor max retries ({gpt_settings.max_supervisor_retries}) exceeded; returning last answer."
            )
            return response, new_messages

        feedback_message = Message(role="user", content=f"Please correct your previous response: {reason}")
        retry_messages = full_history + [feedback_message]
        return await self.get_chat_response(
            messages=retry_messages,
            user=user,
            model=model,
            system_prompt=system_prompt,
            interface=interface,
            supervisor_retry_count=supervisor_retry_count + 1,
        )

    async def get_available_models(self, image_generation: bool = False) -> list[ModelChangeSchema]:
        raise NotImplementedError

    def get_model_display_name(self, model_name: str) -> str:
        return model_name.replace("-", " ").title()

    async def transcribe(self, audio: BytesIO, model: str | None = None) -> str:
        raise NotImplementedError

    async def speech(self, text: str, voice: str | None = None, model: str | None = None) -> bytes:
        raise NotImplementedError

    async def moderate_command(self, cmd: str, model: str | None = None) -> ModeratorsAnswer:
        raise NotImplementedError

    async def supervise(self, context: str, model: str | None = None) -> SupervisorAnswer:
        raise NotImplementedError

    async def vision(
        self, image: bytes, mime_type: str, model: str | None = None, prompt: str | None = None
    ) -> VisionResultSchema:
        raise NotImplementedError

    async def api_key_is_valid(self) -> bool:
        try:
            await self.get_available_models()
        except Exception:  # Some providers return 403, others - 400... Okay..
            return False
        return True

    async def ocr(self, pdf: bytes, model: str | None = None) -> VisionResultSchema:
        raise NotImplementedError

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

    @staticmethod
    def adapt_tool_for_responses(tool: dict | ChatCompletionFunctionToolParam) -> dict[str, Any]:
        """Convert Chat Completions tool to Responses API format.

        Transforms nested function tool definition:
            {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
        To flat Responses API format:
            {"type": "function", "name": ..., "description": ..., "parameters": ..., "strict": False}

        Non-function tools pass through unchanged.

        Args:
            tool: Tool definition in Chat Completions format.

        Returns:
            Tool definition in Responses API format.
        """
        tool_dict = dict(tool)
        if tool_dict.get("type") != "function":
            return tool_dict

        func = tool_dict.get("function", {})
        return {
            "type": "function",
            "name": func.get("name"),
            "description": func.get("description"),
            "parameters": func.get("parameters"),
            "strict": False,
        }

    async def get_images(self, prompt: str, model: str | None) -> list[str] | list[BytesIO]:
        raise NotImplementedError

    def _get_max_tokens_value(self, model_name: str) -> int:
        return getattr(self, "max_tokens", gpt_settings.max_tokens)

    def _get_temperature_value(self, model_name: str) -> float | OpenAIOmit:
        return getattr(self, "temperature", gpt_settings.temperature)

    async def call_functions(
        self,
        calls: list[ToolCallSchema],
        caller_model: str,
        caller_provider: str,
        messages: list[Message],
        system_prompt: str,
        user_id: int | None = None,
        interface: UserInterface | None = None,
    ) -> list[ToolResponseSchema]:
        """Execute a batch of tool calls with optional supervisor pre-check.

        When ``gpt_settings.supervisor_enabled`` is True, each requested tool
        call is evaluated by the Supervisor before the real tool is invoked.
        Calls that receive an ``intervene`` verdict are skipped and produce a
        ``ToolResponseSchema(status="error", result=reason)``. Calls that
        receive ``ok`` are executed normally. Supervisor checks for parallel
        tool calls run concurrently via ``asyncio.gather``.

        Args:
            calls: Tool calls requested by the model.
            caller_model: Name of the model that requested the tools.
            caller_provider: Name of the provider that hosts the caller model.
            messages: Canonical conversation history visible to the caller
                model (used as Supervisor context).
            system_prompt: Already-prepared system prompt that was sent to the
                caller model (used as Supervisor context).
            user_id: Optional ID of the user who triggered the request.
            interface: Optional interface for sending progress/thoughts.

        Returns:
            A list of ``ToolResponseSchema`` results, one per requested call,
            in the same order as ``calls``.
        """
        tool_context: dict[str, Any] = {
            "user_id": user_id,
            "interface": interface,
            "caller_model": caller_model,
            "caller_provider": caller_provider,
        }

        supervisor = None
        if gpt_settings.supervisor_enabled:
            supervisor = RegisteredProviders().first_supervisor_ready
            if supervisor is None:
                logger.warning(
                    "Supervisor is enabled but no supervisor provider could be resolved; running without it."
                )

        if supervisor is not None:
            supervise_coroutines = [
                supervisor.supervise(
                    context=build_supervisor_context(
                        system_prompt=system_prompt,
                        messages=messages,
                        tool_call=SupervisorToolCallAction(tool_name=call.tool_name, args=call.args),
                    )
                )
                for call in calls
            ]
            supervise_results: list[SupervisorAnswer] = await asyncio.gather(*supervise_coroutines)

            allowed_calls: list[ToolCallSchema] = []
            allowed_indices: list[int] = []
            blocked_responses: dict[int, ToolResponseSchema] = {}

            for index, (call, supervise_result) in enumerate(zip(calls, supervise_results)):
                if supervise_result.verdict == SupervisorVerdict.INTERVENE:
                    reason = supervise_result.reason or "Supervisor intervened."
                    blocked_responses[index] = ToolResponseSchema(
                        tool_name=call.tool_name,
                        status="error",
                        result=reason,
                    )
                    logger.warning(f"Supervisor intervened on {call.tool_name}: {reason}")
                else:
                    allowed_calls.append(call)
                    allowed_indices.append(index)

            if allowed_calls:
                tool_coroutines = [
                    RegisteredChibiTools.call(tool_name=call.tool_name, tools_args=tool_context | call.args)
                    for call in allowed_calls
                ]
                allowed_results = await asyncio.gather(*tool_coroutines)
            else:
                allowed_results = []

            results: list[ToolResponseSchema] = [
                ToolResponseSchema(tool_name=call.tool_name, status="pending", result="") for call in calls
            ]
            for allowed_index, result in zip(allowed_indices, allowed_results):
                results[allowed_index] = result
            for blocked_index, blocked_response in blocked_responses.items():
                results[blocked_index] = blocked_response

            return results

        tool_coroutines = [
            RegisteredChibiTools.call(tool_name=call.tool_name, tools_args=tool_context | call.args) for call in calls
        ]
        results = await asyncio.gather(*tool_coroutines)
        return results

    def filter_and_return_list_of_models(
        self, models: list[ModelChangeSchema], image_generation: bool = False
    ) -> list[ModelChangeSchema]:
        all_models = sorted(models, key=lambda model: model.name, reverse=True)

        if image_generation:
            filtered_models = [model for model in all_models if model.image_generation]
        else:
            filtered_models = [model for model in all_models if self.is_chat_ready_model(model.name)]

        if gpt_settings.models_whitelist:
            return [model for model in filtered_models if model.name in gpt_settings.models_whitelist]

        if gpt_settings.models_blacklist:
            return [model for model in filtered_models if model.name not in gpt_settings.models_blacklist]

        return filtered_models


class OpenAIFriendlyProvider(Provider, Generic[P, R]):
    temperature: float | OpenAINotGiven | None = gpt_settings.temperature
    max_tokens: int | OpenAINotGiven | None = gpt_settings.max_tokens
    presence_penalty: float | OpenAINotGiven | None = gpt_settings.presence_penalty
    frequency_penalty: float | OpenAIOmit | None = gpt_settings.frequency_penalty
    image_quality: Literal["standard", "hd", "low", "medium", "high", "auto"] | OpenAIOmit = gpt_settings.image_quality
    image_size: IMAGE_SIZE_OPENAI_LITERAL | None = gpt_settings.image_size_openai
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

    @client.setter
    def client(self, value: AsyncOpenAI) -> None:
        """Setter for client property to allow mocking in tests."""
        # Store the mock value in the instance __dict__ to bypass the property getter
        self.__dict__["_mock_client"] = value

    def get_client(self) -> AsyncOpenAI:
        """Get the client, checking for mock first."""
        if "_mock_client" in self.__dict__:
            return self.__dict__["_mock_client"]
        return self.client

    async def _get_chat_response_impl(
        self,
        messages: list[Message],
        user: User,
        model: str | None = None,
        system_prompt: str = gpt_settings.assistant_prompt,
        interface: UserInterface | None = None,
    ) -> tuple[ChatResponseSchema, list[Message]]:
        """OpenAI-friendly chat completion implementation.

        Args:
            messages: Conversation history in canonical format.
            user: The user requesting the response.
            model: Optional model override.
            system_prompt: Base system prompt template.
            interface: Optional interface for progress/thoughts.

        Returns:
            A tuple of the chat response and the list of new messages
            produced by this call.
        """
        model = model or self.default_model

        initial_messages = [msg.to_openai() for msg in messages]
        chat_response, updated_messages = await self._get_chat_completion_response(
            messages=initial_messages.copy(),
            original_messages=list(messages),
            model=model,
            system_prompt=system_prompt,
            user=user,
            interface=interface,
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
        user: User,
        original_messages: list[Message],
        system_prompt: str | None = None,
        interface: UserInterface | None = None,
    ) -> tuple[ChatResponseSchema, list[ChatCompletionMessageParam]]:
        dialog: list[ChatCompletionMessageParam]
        prepared_system_prompt = ""
        if not system_prompt:
            dialog = messages
        else:
            prepared_system_prompt = await prepare_system_prompt(
                base_system_prompt=system_prompt, user_id=user.id, interface=interface
            )
            system_message = ChatCompletionSystemMessageParam(role="system", content=prepared_system_prompt)
            dialog = [system_message] + messages

        response: ChatCompletion = await self.client.chat.completions.create(  # type: ignore
            model=model,
            messages=dialog,
            temperature=self._get_temperature_value(model_name=model),
            max_tokens=self._get_max_tokens_value(model_name=model),
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            timeout=self.timeout,
            tools=RegisteredChibiTools.get_tool_definitions(),
            tool_choice="auto",
            reasoning_effort=self.get_reasoning_effort_value(model_name=model),
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

        tool_calls: list[ChatCompletionMessageToolCall] | None = data.message.tool_calls  # type: ignore

        if not tool_calls:
            messages.append(ChatCompletionAssistantMessageParam(**data.message.model_dump()))  # type: ignore
            original_messages.append(Message(role="assistant", content=answer))
            return ChatResponseSchema(answer=answer, provider=self.name, model=model, usage=usage), messages

        # Tool calls handling
        logger.log("CALL", f"{model} requested the call of {len(tool_calls)} tools.")

        thoughts = answer or "No thoughts"
        if answer:
            await send_llm_thoughts(thoughts=thoughts, interface=interface)
        logger.log("THINK", f"{model}: {thoughts}. {usage_message}")

        calls = [
            ToolCallSchema(
                tool_name=tool_call.function.name,
                args=json.loads(tool_call.function.arguments),
            )
            for tool_call in tool_calls
        ]
        results = await self.call_functions(
            calls=calls,
            caller_model=model,
            caller_provider=self.name,
            messages=original_messages,
            system_prompt=prepared_system_prompt,
            user_id=user.id,
            interface=interface,
        )

        for tool_call, result in zip(tool_calls, results):
            # Temporary hotfix: preserve reasoning_content for DeepSeek/Moonshot thinking mode
            message_dict: dict[str, Any] = {
                "role": "assistant",
                "content": answer,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                ],
            }
            # Add reasoning_content if present (DeepSeek-Reasoner, Moonshot KIMI, etc.)
            if hasattr(data.message, "reasoning_content") and data.message.reasoning_content:
                logger.log("THINK", data.message.reasoning_content)
                message_dict["reasoning_content"] = data.message.reasoning_content

            messages.append(message_dict)  # type: ignore
            original_messages.append(
                Message(
                    role="assistant",
                    content=answer,
                    tool_calls=[
                        ToolSchema(
                            id=tool_call.id,
                            type="function",
                            function=FunctionSchema(
                                name=tool_call.function.name,
                                arguments=tool_call.function.arguments,
                            ),
                        )
                    ],
                )
            )
            tool_result_message = ChatCompletionToolMessageParam(
                tool_call_id=tool_call.id,
                role="tool",
                content=result.model_dump_json(),
            )
            messages.append(tool_result_message)
            original_messages.append(
                Message(
                    role="tool",
                    content=result.model_dump_json(),
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.function.name,
                )
            )

        logger.log("CALL", "All the function results have been obtained. Returning them to the LLM...")
        return await self._get_chat_completion_response(
            messages=messages,
            original_messages=original_messages,
            model=model,
            user=user,
            system_prompt=system_prompt,
            interface=interface,
        )

    def get_reasoning_effort_value(self, model_name: str) -> ReasoningEffort | OpenAIOmit | None:
        return omit

    async def _classify_with_text(
        self,
        system_prompt: str,
        response_model: type[T],
        user_content: str,
        model: str | None = None,
    ) -> T:
        """Run a structured classification via an OpenAI chat completion.

        The LLM is asked to emit a JSON object matching ``response_model``.
        The raw text is stripped of markdown/JSON wrappers before parsing.

        Args:
            system_prompt: System instruction for the classifier.
            response_model: Pydantic model used to validate the JSON response.
            user_content: User message content to classify.
            model: Optional moderator model override.

        Returns:
            Validated instance of ``response_model``.

        Raises:
            ServiceResponseError: If the LLM returns an empty choices list.
            ValueError: If the response cannot be parsed or validated.
        """
        moderator_model = model or self.default_moderation_model or self.default_model
        system_message = ChatCompletionSystemMessageParam(role="system", content=system_prompt)
        messages = [Message(role="user", content=user_content).to_openai()]
        dialog: list[ChatCompletionMessageParam] = [system_message] + messages

        temperature = (
            1 if moderator_model.startswith("o") or "mini" in moderator_model or "nano" in moderator_model else 0.0
        )
        response: ChatCompletion = await self.client.chat.completions.create(
            model=moderator_model,
            messages=dialog,
            temperature=temperature,
            max_completion_tokens=1024,
            presence_penalty=cast(float | OpenAIOmit | None, self.presence_penalty),
            frequency_penalty=self.frequency_penalty,
            timeout=self.timeout,
            reasoning_effort=self.get_reasoning_effort_value(model_name=moderator_model),
        )

        choices: list[Choice] = response.choices
        if len(choices) == 0:
            raise ServiceResponseError(
                provider=self.name, model=moderator_model, detail="Unexpected (empty) response received"
            )

        answer: str = choices[0].message.content or ""

        usage = get_usage_from_openai_response(response_message=response)
        if application_settings.is_influx_configured:
            MetricsService.send_usage_metrics(metric=usage, model=moderator_model, provider=self.name)

        answer = answer.strip("`").strip("json").strip()
        try:
            result_data = json.loads(answer)
        except Exception as e:
            raise ValueError(answer) from e

        try:
            return response_model.model_validate(result_data)
        except Exception as e:
            raise ValueError(f"Error parsing moderator's response: {answer}. Error: {e}") from e

    async def moderate_command(self, cmd: str, model: str | None = None) -> ModeratorsAnswer:
        """Moderate a command using the OpenAI-compatible moderation flow.

        Args:
            cmd: The command string to evaluate.
            model: Optional moderator model override.

        Returns:
            A ``ModeratorsAnswer`` with the moderation verdict.
        """
        try:
            result = await self._classify_with_text(
                system_prompt=MODERATOR_PROMPT,
                response_model=ModeratorsAnswer,
                user_content=cmd,
                model=model,
            )
        except ServiceResponseError:
            raise
        except Exception as e:
            logger.error(f"Error parsing moderator's response: {e}")
            return ModeratorsAnswer(verdict="declined", reason=str(e), status="error")

        if result.verdict == "accepted":
            return ModeratorsAnswer(verdict="accepted", status="ok")

        if result.reason is None:
            logger.error(f"Moderator did not provide reason properly: {cmd}")

        return ModeratorsAnswer(verdict="declined", reason=result.reason, status="operation aborted")

    async def supervise(self, context: str, model: str | None = None) -> SupervisorAnswer:
        """Evaluate an agent action for role/flow compliance (OpenAI-text path).

        Uses the same text-based classification helper as ``moderate_command``
        but with the ``SUPERVISOR_PROMPT`` and ``SupervisorAnswer`` schema.

        Args:
            context: Full serialized context for the supervisor to evaluate.
            model: Optional supervisor model override.

        Returns:
            A ``SupervisorAnswer`` with the compliance verdict. On any failure
            the method returns ``SupervisorAnswer(verdict=OK, status="error")``
            (fail-open policy).

        Raises:
            Does NOT raise exceptions -- all failures are caught and surfaced
            via the returned ``SupervisorAnswer``.
        """
        try:
            resolved_model = model or gpt_settings.supervisor_model_resolved
            return await self._classify_with_text(
                system_prompt=SUPERVISOR_PROMPT,
                response_model=SupervisorAnswer,
                user_content=context,
                model=resolved_model,
            )
        except Exception as e:
            logger.error(f"Supervisor error in {self.name}: {e}")
            return SupervisorAnswer(verdict=SupervisorVerdict.OK, status="error")

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
        return self.filter_and_return_list_of_models(models=all_models, image_generation=image_generation)

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

        images: list[Image] = response.data

        if response.data[0].url:
            return [image.url for image in images if image.url]

        return [BytesIO(base64.b64decode(image.b64_json)) for image in images if image.b64_json]

    async def vision(
        self,
        image: bytes,
        mime_type: str,
        model: str | None = None,
        prompt: str | None = None,
    ) -> VisionResultSchema:
        model = model or self.default_vision_model
        if not model:
            raise NoModelSelectedError(provider=self.name, detail="No vision model selected")
        prompt = prompt or "Describe the image in detail."
        logger.info(f"[{self.name}] Analyzing image with model {model}...")

        # Encode image to base64
        image_base64 = base64.b64encode(image).decode("utf-8")
        data_url = f"data:{mime_type};base64,{image_base64}"

        # Use parse() for structured output with Pydantic models
        response = await self.get_client().chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url, "detail": "high"},
                        },
                    ],
                }
            ],
            response_format=VisionResultSchema,
            max_tokens=4096,
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

        logger.info(f"[{self.name}] Image analyzed successfully: {result.parsed.short_description}...")
        return result.parsed

    async def transcribe(self, audio: BytesIO, model: str | None = None) -> str:
        model = model or self.stt_model
        logger.info(f"[{self.name}] Transcribing audio with model {model}...")
        response = await self.client.audio.transcriptions.create(
            model=model,
            file=("voice.ogg", audio.getvalue()),
        )
        if response:
            logger.info(f"[{self.name}] Transcribed text: {response.text}")
            return response.text
        raise ValueError("Could not transcribe audio message")

    async def speech(self, text: str, voice: str | None = None, model: str | None = None) -> bytes:
        voice = voice or self.tts_voice
        model = model or self.tts_model
        logger.info(f"[{self.name}] Recording a voice message with model {model}...")
        response = await self.client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
        )
        return await response.aread()


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
    base_url: str = "https://api.anthropic.com"

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

    async def _get_chat_response_impl(
        self,
        messages: list[Message],
        user: User,
        model: str | None = None,
        system_prompt: str = gpt_settings.assistant_prompt,
        interface: UserInterface | None = None,
    ) -> tuple[ChatResponseSchema, list[Message]]:
        """Anthropic-friendly chat completion implementation.

        Args:
            messages: Conversation history in canonical format.
            user: The user requesting the response.
            model: Optional model override.
            system_prompt: Base system prompt template.
            interface: Optional interface for progress/thoughts.

        Returns:
            A tuple of the chat response and the list of new messages
            produced by this call.
        """
        model = model or self.default_model
        initial_messages = [msg.to_anthropic() for msg in messages]

        if len(initial_messages) >= 2:
            initial_messages[-2]["content"][0]["cache_control"] = {"type": "ephemeral"}  # type: ignore

        chat_response, updated_messages = await self._get_chat_completion_response(
            messages=initial_messages.copy(),
            original_messages=list(messages),
            user=user,
            model=model,
            system_prompt=system_prompt,
            interface=interface,
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
        user: User,
        original_messages: list[Message],
        system_prompt: str = gpt_settings.assistant_prompt,
        interface: UserInterface | None = None,
    ) -> tuple[ChatResponseSchema, list[MessageParam]]:
        prepared_system_prompt = await prepare_system_prompt(
            base_system_prompt=system_prompt, user_id=user.id, interface=interface
        )
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

            original_messages.append(Message(role="assistant", content=answer or "no data"))
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
            await send_llm_thoughts(thoughts=thoughts_part.text, interface=interface)

        logger.log(
            "THINK", f"{model}: {thoughts_part.text if thoughts_part else 'No thoughts'}. {get_usage_msg(usage=usage)}"
        )

        calls = [
            ToolCallSchema(
                tool_name=tool_call_part.name,
                args=tool_call_part.input,
            )
            for tool_call_part in tool_call_parts
        ]
        results = await self.call_functions(
            calls=calls,
            caller_model=model,
            caller_provider=self.name,
            messages=original_messages,
            system_prompt=prepared_system_prompt,
            user_id=user.id,
            interface=interface,
        )

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

            original_messages.append(
                Message(
                    role="assistant",
                    content=thoughts_part.text if thoughts_part else "",
                    tool_calls=[
                        ToolSchema(
                            id=tool_call_part.id,
                            type="tool_use",
                            function=FunctionSchema(
                                name=tool_call_part.name,
                                arguments=json.dumps(tool_call_part.input),
                            ),
                        )
                    ],
                )
            )
            original_messages.append(
                Message(
                    role="tool",
                    content=result.model_dump_json(),
                    tool_call_id=tool_call_part.id,
                    tool_name=tool_call_part.name,
                )
            )

        logger.log("CALL", "All the function results have been obtained. Returning them to the LLM...")
        return await self._get_chat_completion_response(
            messages=messages,
            original_messages=original_messages,
            model=model,
            user=user,
            system_prompt=system_prompt,
            interface=interface,
        )

    async def _classify_with_forced_tool(
        self,
        system_prompt: str,
        response_model: type[T],
        user_content: str,
        tool_name: str,
        model: str | None = None,
    ) -> T:
        """Classify content by forcing an Anthropic tool call.

        The model is instructed to call ``tool_name``. If it ignores the forced
        tool choice and returns plain text, a JSON object is extracted from the
        text and validated against ``response_model`` as a fallback.

        Args:
            system_prompt: System instruction for the classifier.
            response_model: Pydantic model used to validate the tool input.
            user_content: User message content to classify.
            tool_name: Name of the forced tool to call.
            model: Optional moderator model override.

        Returns:
            Validated instance of ``response_model``.

        Raises:
            ValueError: If the response is empty, contains no usable tool input,
                or the input cannot be parsed/validated.
        """
        moderator_model = model or self.default_moderation_model or self.default_model
        messages = [Message(role="user", content=user_content).to_anthropic()]
        moderator_prompt = f"{system_prompt}\n**HARD RULE:** call the {tool_name} tool to provide your verdict"

        schema = response_model.model_json_schema()
        response_message: AnthropicMessage = await self.client.messages.create(
            model=moderator_model,
            max_tokens=1024,
            temperature=0.1,
            timeout=self.timeout,
            system=[TextBlockParam(text=moderator_prompt, type="text")],
            tools=[
                ToolParam(
                    name=tool_name,
                    description="Provide the classifier verdict via calling this tool.",
                    input_schema=InputSchemaTyped(
                        type="object",
                        properties=cast(dict[str, object], schema.get("properties", {})),
                        required=cast(list[str], schema.get("required", [])),
                    ),
                )
            ],
            tool_choice=ToolChoiceToolParam(type="tool", name=tool_name),
            messages=messages,
        )
        if not response_message.content:
            raise ValueError("no response from moderator received")

        usage = get_usage_from_anthropic_response(response_message=response_message)
        if application_settings.is_influx_configured:
            MetricsService.send_usage_metrics(metric=usage, model=moderator_model, provider=self.name)

        tool_call: ToolUseBlock | None = next(
            (part for part in response_message.content if isinstance(part, ToolUseBlock)), None
        )
        if tool_call is not None:
            try:
                return response_model.model_validate(tool_call.input, extra="ignore")
            except Exception as e:
                raise ValueError(f"Error parsing moderator's response: {tool_call.input}. Error: {e}") from e

        text_part: TextBlock | None = next(
            (part for part in response_message.content if isinstance(part, TextBlock)), None
        )
        if text_part is None or not text_part.text.strip():
            raise ValueError("no response from moderator received")

        raw_text = text_part.text.strip()
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match is None:
            raise ValueError(f"Moderator returned no tool call and no JSON verdict. Raw: {raw_text[:200]}")

        try:
            parsed = json.loads(match.group(0))
            return response_model.model_validate(parsed, extra="ignore")
        except Exception as e:
            raise ValueError(f"Error parsing moderator's text verdict: {raw_text[:200]}. Error: {e}") from e

    async def moderate_command(self, cmd: str, model: str | None = None) -> ModeratorsAnswer:
        """Moderate a command using the Anthropic-compatible moderation flow.

        Args:
            cmd: The command string to evaluate.
            model: Optional moderator model override.

        Returns:
            A ``ModeratorsAnswer`` with the moderation verdict.
        """
        try:
            result = await self._classify_with_forced_tool(
                system_prompt=MODERATOR_PROMPT,
                response_model=ModeratorsAnswer,
                user_content=cmd,
                tool_name="print_moderator_verdict",
                model=model,
            )
        except Exception as e:
            logger.error(str(e))
            return ModeratorsAnswer(verdict="declined", reason=str(e), status="error")

        if result.verdict == "accepted":
            return ModeratorsAnswer(verdict="accepted", status="ok")
        return ModeratorsAnswer(verdict="declined", reason=result.reason, status="operation aborted")

    async def supervise(self, context: str, model: str | None = None) -> SupervisorAnswer:
        """Evaluate an agent action for role/flow compliance (Anthropic-tool path).

        Uses the same forced-tool classification helper as ``moderate_command``
        but with the ``SUPERVISOR_PROMPT`` and ``SupervisorAnswer`` schema.

        Args:
            context: Full serialized context for the supervisor to evaluate.
            model: Optional supervisor model override.

        Returns:
            A ``SupervisorAnswer`` with the compliance verdict. On any failure
            the method returns ``SupervisorAnswer(verdict=OK, status="error")``
            (fail-open policy).

        Raises:
            Does NOT raise exceptions -- all failures are caught and surfaced
            via the returned ``SupervisorAnswer``.
        """
        try:
            resolved_model = model or gpt_settings.supervisor_model_resolved
            return await self._classify_with_forced_tool(
                system_prompt=SUPERVISOR_PROMPT,
                response_model=SupervisorAnswer,
                user_content=context,
                tool_name="print_supervisor_verdict",
                model=resolved_model,
            )
        except Exception as e:
            logger.error(f"Supervisor error in {self.name}: {e}")
            return SupervisorAnswer(verdict=SupervisorVerdict.OK, status="error")

    async def get_available_models(self, image_generation: bool = False) -> list[ModelChangeSchema]:
        if image_generation:
            return []

        try:
            response = await self._request(method="GET", url=urljoin(self.base_url, "v1/models"))
        except Exception as e:
            logger.error(f"Failed to get available models for provider {self.name} due to exception: {e}")
            return []

        response_data = response.json().get("data", [])
        all_models = [
            ModelChangeSchema(
                provider=self.name,
                name=model.get("id"),
                display_name=model.get("display_name") or model.get("id"),
                image_generation=False,
            )
            for model in response_data
            if model.get("id") and (model.get("type") == "model" or model.get("object") == "model")
        ]
        return self.filter_and_return_list_of_models(models=all_models, image_generation=image_generation)
