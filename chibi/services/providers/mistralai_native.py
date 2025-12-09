import asyncio
import json
import random
from asyncio import sleep
from typing import Any, Union

from loguru import logger
from mistralai import ChatCompletionResponse, Mistral, TextChunk
from mistralai.models import (
    AssistantMessage,
    FunctionCall,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from openai.types.chat import ChatCompletionToolParam
from telegram import Update
from telegram.ext import ContextTypes

from chibi.config import application_settings, gpt_settings
from chibi.exceptions import NoApiKeyProvidedError, NoResponseError
from chibi.models import Message, User
from chibi.schemas.app import ChatResponseSchema, ModelChangeSchema
from chibi.services.metrics import MetricsService
from chibi.services.providers.provider import RestApiFriendlyProvider
from chibi.services.providers.tools import RegisteredChibiTools, tools
from chibi.services.providers.utils import (
    get_usage_from_mistral_response,
    get_usage_msg,
    prepare_system_prompt,
    send_llm_thoughts,
)

MistralMessageParam = Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage]


class MistralAI(RestApiFriendlyProvider):
    api_key = gpt_settings.mistralai_key
    chat_ready = True

    name = "MistralAI"
    model_name_keywords = ["mistral", "mixtral", "ministral"]
    model_name_keywords_exclude = ["embed", "moderation", "ocr"]
    default_model = "mistral-medium-latest"
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
        return tools

    @property
    def client(self) -> Mistral:
        if self._client:
            return self._client

        if not self.token:
            raise NoApiKeyProvidedError(provider=self.name)

        self._client = Mistral(api_key=self.token)
        return self._client

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
    ) -> ChatCompletionResponse:
        """Generate content with retry logic."""
        for attempt in range(gpt_settings.retries):
            response = await self.client.chat.complete_async(
                model=model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                tools=self.tools_list,  # type: ignore[arg-type]
                tool_choice="auto",
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
        initial_messages = [msg.to_mistral() for msg in messages]
        chat_response, updated_messages = await self._get_chat_completion_response(
            messages=initial_messages.copy(),
            user=user,
            model=model,
            system_prompt=system_prompt,
            context=context,
            update=update,
        )
        new_messages = [msg for msg in updated_messages if msg not in initial_messages]
        return (
            chat_response,
            [Message.from_mistral(msg) for msg in new_messages if not isinstance(msg, SystemMessage)],
        )

    async def _get_chat_completion_response(
        self,
        messages: list[MistralMessageParam],
        model: str,
        user: User | None = None,
        system_prompt: str = gpt_settings.assistant_prompt,
        update: Update | None = None,
        context: ContextTypes.DEFAULT_TYPE | None = None,
    ) -> tuple[ChatResponseSchema, list[MistralMessageParam]]:
        prepared_system_prompt = await prepare_system_prompt(base_system_prompt=system_prompt, user=user)
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=prepared_system_prompt, role="system")] + messages
        else:
            messages = [SystemMessage(content=prepared_system_prompt, role="system")] + messages[1:]

        response: ChatCompletionResponse = await self._generate_content(model=model, messages=messages)
        usage = get_usage_from_mistral_response(response_message=response)

        if application_settings.is_influx_configured:
            MetricsService.send_usage_metrics(metric=usage, user=user, model=model, provider=self.name)

        message_data = response.choices[0].message

        tool_calls = message_data.tool_calls
        if not tool_calls:
            messages.append(
                AssistantMessage(
                    content=message_data.content or "",
                )
            )
            return ChatResponseSchema(
                answer=message_data.content or "no data",
                provider=self.name,
                model=model,
                usage=usage,
            ), messages

        # Tool calls handling
        logger.log("CALL", f"LLM requested the call of {len(tool_calls)} tools.")

        thoughts = self.get_thoughts(assistant_message=message_data)
        if thoughts:
            await send_llm_thoughts(thoughts=thoughts, context=context, update=update)

        logger.log("THINK", f"{model}: {thoughts}. {get_usage_msg(usage=usage)}")
        tool_context: dict[str, Any] = {
            "user_id": user.id if user else None,
            "telegram_context": context,
            "telegram_update": update,
            "model": model,
        }

        tool_coroutines = []
        for tool_call in tool_calls:
            function_args = (
                json.loads(tool_call.function.arguments)
                if isinstance(tool_call.function.arguments, str)
                else tool_call.function.arguments
            )
            tool_coroutines.append(
                RegisteredChibiTools.call(tool_name=tool_call.function.name, tools_args=tool_context | function_args)
            )
        results = await asyncio.gather(*tool_coroutines)

        for tool_call, result in zip(tool_calls, results):
            tool_call_message = AssistantMessage(
                content=message_data.content or "",
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

        logger.log("CALL", "The all functions results have been obtained. Returning them to the LLM...")
        return await self._get_chat_completion_response(
            messages=messages,
            model=model,
            user=user,
            system_prompt=system_prompt,
            context=context,
            update=update,
        )

    async def get_available_models(self, image_generation: bool = False) -> list[ModelChangeSchema]:
        if image_generation:
            return []

        try:
            response = await self.client.models.list_async()
            mistral_models = response.data or []
        except Exception as e:
            logger.error(f"Failed to get available models for provider {self.name} due to exception: {e}")
            return []

        all_models = [
            ModelChangeSchema(
                provider=self.name,
                name=model.id,
                display_name=model.id.replace("-", " ").title(),
                image_generation=False,
            )
            for model in mistral_models
        ]
        all_models.sort(key=lambda model: model.name)

        if gpt_settings.models_whitelist:
            return [model for model in all_models if model.name in gpt_settings.models_whitelist]

        return all_models
