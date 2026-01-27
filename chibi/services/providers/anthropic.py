import asyncio
import random
from asyncio import sleep
from typing import Any

from anthropic import AsyncClient, NotGiven, Omit
from anthropic.types import (
    CacheControlEphemeralParam,
    MessageParam,
    TextBlock,
    TextBlockParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlock,
)
from anthropic.types import (
    Message as AnthropicMessage,
)
from loguru import logger
from telegram import Update
from telegram.ext import ContextTypes

from chibi.config import application_settings, gpt_settings
from chibi.exceptions import NoApiKeyProvidedError, NoResponseError
from chibi.models import Message, User
from chibi.schemas.app import ChatResponseSchema, ModelChangeSchema
from chibi.services.metrics import MetricsService
from chibi.services.providers.provider import RestApiFriendlyProvider
from chibi.services.providers.tools import RegisteredChibiTools
from chibi.services.providers.utils import (
    get_usage_from_anthropic_response,
    get_usage_msg,
    prepare_system_prompt,
    send_llm_thoughts,
)


class Anthropic(RestApiFriendlyProvider):
    api_key = gpt_settings.anthropic_key
    chat_ready = True

    name = "Anthropic"
    model_name_keywords = ["claude"]
    default_model = "claude-sonnet-4-5-20250929"
    frequency_penalty: float | NotGiven | None = gpt_settings.frequency_penalty
    max_tokens: int = gpt_settings.max_tokens
    presence_penalty: float | NotGiven = gpt_settings.presence_penalty
    temperature: float | Omit = gpt_settings.temperature

    def __init__(self, token: str) -> None:
        self._client: AsyncClient | None = None
        super().__init__(token=token)

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
        if self._client:
            return self._client

        if not self.token:
            raise NoApiKeyProvidedError(provider=self.name)

        self._client = AsyncClient(api_key=self.token)
        return self._client

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "x-api-key": self.token,
            "anthropic-version": "2023-06-01",
        }

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
        initial_messages = [msg.to_anthropic() for msg in messages]

        if len(initial_messages) >= 2:
            initial_messages[1]["content"][0]["cache_control"] = {"type": "ephemeral"}  # type: ignore
            initial_messages[-2]["content"][0]["cache_control"] = {"type": "ephemeral"}  # type: ignore

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
            [Message.from_anthropic(msg) for msg in new_messages],
        )

    async def _get_chat_completion_response(
        self,
        messages: list[MessageParam],
        model: str,
        user: User | None = None,
        system_prompt: str = gpt_settings.assistant_prompt,
        update: Update | None = None,
        context: ContextTypes.DEFAULT_TYPE | None = None,
    ) -> tuple[ChatResponseSchema, list[MessageParam]]:
        prepared_system_prompt = await prepare_system_prompt(base_system_prompt=system_prompt, user=user)
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
            return ChatResponseSchema(
                answer=getattr(response_message.content[0], "text", "no data"),
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
            await send_llm_thoughts(thoughts=thoughts_part.text, context=context, update=update)

        logger.log(
            "THINK", f"{model}: {thoughts_part.text if thoughts_part else 'No thoughts'}. {get_usage_msg(usage=usage)}"
        )

        tool_context: dict[str, Any] = {
            "user_id": user.id if user else None,
            "telegram_context": context,
            "telegram_update": update,
            "model": model,
        }

        tool_coroutines = [
            RegisteredChibiTools.call(tool_name=tool_call_part.name, tools_args=tool_context | tool_call_part.input)
            for tool_call_part in tool_call_parts
        ]
        results = await asyncio.gather(*tool_coroutines)

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

        logger.log("CALL", "All the function results have been obtained. Returning them to the LLM...")
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

        url = "https://api.anthropic.com/v1/models"
        try:
            response = await self._request(method="GET", url=url)
        except Exception as e:
            logger.error(f"Failed to get available models for provider {self.name} due to exception: {e}")
            return []

        response_data = response.json().get("data", [])
        all_models = [
            ModelChangeSchema(
                provider=self.name,
                name=model.get("id"),
                display_name=model.get("display_name"),
                image_generation=False,
            )
            for model in response_data
            if model.get("id") and model.get("type") == "model"
        ]
        all_models.sort(key=lambda model: model.name)

        if gpt_settings.models_whitelist:
            return [model for model in all_models if model.name in gpt_settings.models_whitelist]

        return all_models
