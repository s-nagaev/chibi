import json
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

from chibi.config import gpt_settings
from chibi.exceptions import NoApiKeyProvidedError, ServiceResponseError
from chibi.models import Message, User
from chibi.schemas.app import ChatResponseSchema, ModelChangeSchema, UsageSchema
from chibi.services.providers.provider import RestApiFriendlyProvider
from chibi.services.providers.tools import registered_functions, tools
from chibi.services.providers.tools.schemas import ToolResponse
from chibi.services.providers.utils import (
    escape_and_truncate,
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
            for tool in tools
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
        raise ServiceResponseError(provider=self.name, model=model, detail="Unexpected (empty) response received")

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
        prepared_system_prompt = (
            await prepare_system_prompt(base_system_prompt=system_prompt, user=user) if user else system_prompt
        )
        response_message = await self._generate_content(
            model=model,
            system_prompt=prepared_system_prompt,
            messages=messages,
        )

        tool_call_parts = [part for part in response_message.content if isinstance(part, ToolUseBlock)]

        thinking_text_part = next(
            (part for part in response_message.content if isinstance(part, TextBlock)),
            TextBlock(type="text", text="No thoughts..."),
        )

        usage = get_usage_from_anthropic_response(response_message=response_message)
        usage_message = get_usage_msg(usage=usage)
        function_response: ToolResponse

        for part in tool_call_parts:
            function_to_call = registered_functions.get(part.name)
            if not function_to_call:
                logger.error(f"Function {part.name} called but it's not registered.")
                function_to_call = registered_functions["stub_function"]

            if all((context, update, thinking_text_part.text != "No thoughts...")):
                await send_llm_thoughts(thoughts=thinking_text_part.text, context=context, update=update)

            logger.log("THINK", f"{model}: {thinking_text_part.text}. {usage_message}")

            function_args: dict[str, Any] = {
                "user_id": user.id if user else None,
                "telegram_context": context,
                "telegram_update": update,
            }

            function_args.update(part.input)
            logger.log("CALL", f"Calling a function '{part.name}'. Args: {escape_and_truncate(json.dumps(part.input))}")

            function_response = await function_to_call(**function_args)

            tool_call_message = MessageParam(
                role="assistant",
                content=[thinking_text_part.model_dump(), part.model_dump()],  # type: ignore
            )
            tool_result_message = MessageParam(
                role="user",
                content=[
                    ToolResultBlockParam(
                        type="tool_result",
                        tool_use_id=part.id,
                        content=function_response.model_dump_json(),
                    )
                ],
            )
            messages.append(tool_call_message)
            messages.append(tool_result_message)
            logger.log("CALL", f"Function result received, returning it to the model {model}...")

            return await self._get_chat_completion_response(
                messages=messages,
                model=model,
                user=user,
                system_prompt=system_prompt,
                context=context,
                update=update,
            )

        messages.append(
            MessageParam(
                role="assistant",
                content=[content.model_dump() for content in response_message.content],  # type: ignore
            )
        )
        usage = UsageSchema(
            completion_tokens=response_message.usage.output_tokens,
            prompt_tokens=response_message.usage.input_tokens,
            total_tokens=response_message.usage.output_tokens + response_message.usage.input_tokens,
        )
        return ChatResponseSchema(
            answer=getattr(response_message.content[0], "text", "no data"), provider=self.name, model=model, usage=usage
        ), messages

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
        print(response_data)
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
