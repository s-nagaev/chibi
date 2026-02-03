import asyncio
import base64
import binascii
import itertools
import json
import time
from typing import TYPE_CHECKING, Any, Literal

from anthropic.types import (
    MessageParam,
    TextBlockParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from google.genai.types import ContentDict, FunctionCallDict, PartDict
from mistralai.models import (
    AssistantMessage as MistralAssistantMessage,
)
from mistralai.models import (
    FunctionCall as MistralFunctionCall,
)
from mistralai.models import (
    SystemMessage as MistralSystemMessage,
)
from mistralai.models import (
    ToolCall as MistralToolCall,
)
from mistralai.models import (
    ToolMessage as MistralToolMessage,
)
from mistralai.models import (
    UserMessage as MistralUserMessage,
)
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel, Field, field_serializer, field_validator

from chibi.config import application_settings, gpt_settings
from chibi.exceptions import (
    NoApiKeyProvidedError,
    NoProviderSelectedError,
)
from chibi.schemas.app import ModelChangeSchema

if TYPE_CHECKING:
    from chibi.services.providers import RegisteredProviders
    from chibi.services.providers.provider import Provider

CHAT_COMPLETION_CLASSES = {
    "assistant": ChatCompletionAssistantMessageParam,
    "function": ChatCompletionFunctionMessageParam,
    "tool": ChatCompletionToolMessageParam,
    "user": ChatCompletionUserMessageParam,
}


class FunctionSchema(BaseModel):
    name: str
    arguments: str | None = None


class ToolSchema(BaseModel):
    id: str
    type: str = "function"
    function: FunctionSchema
    thought_signature: bytes | None = None

    @field_validator("thought_signature", mode="before")
    @classmethod
    def decode_signature_from_base64(cls, v: bytes | str | None) -> bytes | None:
        if v is None:
            return None

        if isinstance(v, bytes):
            return v

        if isinstance(v, str):
            try:
                return base64.b64decode(v)
            except binascii.Error:
                raise ValueError("Invalid base64 string for thought_signature")

        raise TypeError("thought_signature must be bytes or a base64 encoded string")

    @field_serializer("thought_signature")
    def serialize_signature_to_base64(self, value: bytes | None) -> str | None:
        if not value:
            return None
        return base64.b64encode(value).decode("ascii")


class Message(BaseModel):
    id: int = Field(default_factory=time.time_ns)
    role: str
    content: str
    expire_at: float | None = None
    tool_calls: list[ToolSchema] | None = None
    tool_call_id: str | None = None
    source: str | None = None

    def to_openai(self) -> ChatCompletionMessageParam:
        wrapper_class = CHAT_COMPLETION_CLASSES.get(self.role)
        if not wrapper_class:
            raise ValueError(f"Role {self.role} seems not supported yet")

        open_ai_message = wrapper_class(**self.model_dump(exclude={"expire_at"}))
        return open_ai_message

    @classmethod
    def from_openai(cls, open_ai_message: ChatCompletionMessageParam) -> "Message":
        # if not open_ai_message.get("tool_calls"):
        #     open_ai_message["tool_calls"] = []
        msg = cls(**open_ai_message)
        msg.source = "openai"
        return msg

    def to_anthropic(self) -> MessageParam:
        if self.role == "tool" and self.tool_call_id:
            return MessageParam(
                role="user",
                content=[
                    ToolResultBlockParam(
                        tool_use_id=self.tool_call_id,
                        type="tool_result",
                        content=self.content,
                    )
                ],
            )
        if self.role == "user":
            return MessageParam(
                role="user",
                content=[
                    TextBlockParam(
                        type="text",
                        text=self.content,
                    )
                ],
            )

        assistant_content: list[TextBlockParam | ToolUseBlockParam] = [
            TextBlockParam(
                type="text",
                text=self.content or "No content",
            )
        ]
        if self.tool_calls:
            for tool in self.tool_calls:
                assistant_content.append(
                    ToolUseBlockParam(
                        type="tool_use",
                        id=tool.id,
                        name=tool.function.name,
                        input=json.loads(tool.function.arguments) if tool.function.arguments else {},
                    )
                )
        return MessageParam(
            role="assistant",
            content=assistant_content,
        )

    @classmethod
    def from_anthropic(cls, anthropic_message: MessageParam) -> "Message":
        message_content = anthropic_message["content"]
        role: Literal["user", "assistant", "tool"] = anthropic_message["role"]
        tool_call_id: str | None = None
        content: str = ""
        tools: list[ToolSchema] = []

        if isinstance(message_content, str):
            return cls(role=role, content=message_content)

        for content_block in message_content:
            if isinstance(content_block, dict):
                # TextBlockParam
                if content_block["type"] == "text":
                    content = content_block.get("text", "No content")

                # ToolResultBlockParam
                if content_block["type"] == "tool_result":
                    role = "tool"
                    tool_call_id = content_block.get("tool_use_id")
                    content_data = content_block["content"]
                    # It is not very clear under what circumstances the content
                    # here might assume the value like `Iterable[Content]`.
                    # In anthropic code (at the moment) there is nothing like
                    # that. It is likely an atavism from the orig openai module
                    content = content_data if isinstance(content_data, str) else "No content"

                # ToolUseBlockParam
                if content_block["type"] == "tool_use":
                    function = FunctionSchema(
                        name=content_block["name"],
                        arguments=json.dumps(content_block["input"]),
                    )
                    tool = ToolSchema(id=content_block["id"], function=function)
                    tools.append(tool)

        return cls(role=role, content=content, tool_calls=tools or None, tool_call_id=tool_call_id, source="anthropic")

    def to_google(self) -> ContentDict:
        """Convert a Chibi Message to a Google AI ContentDict."""

        # Google uses 'model' for the assistant's role
        google_role = "model" if self.role == "assistant" else "user"

        # Handle tool calls from the assistant
        if self.role == "assistant" and self.tool_calls:
            parts: list[PartDict] = []
            # Add text content if present
            if self.content:
                parts.append({"text": self.content})
            # Add function calls
            for tool_call in self.tool_calls:
                args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                parts.append(
                    PartDict(
                        function_call=FunctionCallDict(
                            name=tool_call.function.name,
                            args=args,
                        ),
                        thought_signature=tool_call.thought_signature,
                    )
                )

            return ContentDict(role=google_role, parts=parts)

        # Handle tool responses - Google uses 'user' role for tool responses
        if self.role == "tool" and self.tool_call_id:
            return {
                "role": "user",
                "parts": [
                    {
                        "function_response": {
                            "name": self.tool_call_id,  # This needs to be resolved to function name by provider
                            "response": {"content": self.content},
                        }
                    }
                ],
            }

        # Handle simple text content for user or assistant
        if self.content:
            return {"role": google_role, "parts": [{"text": self.content}]}

        # Return empty content if no other case matches
        return {"role": google_role, "parts": []}

    @classmethod
    def from_google(cls, google_content: ContentDict | dict[str, Any]) -> "Message":
        """Convert a Google AI ContentDict to a Chibi Message."""

        # Map Google role back to Chibi role
        chibi_role = "assistant" if google_content.get("role") == "model" else google_content.get("role", "user")
        content = ""
        tools: list[ToolSchema] = []
        tool_call_id: str | None = None

        parts = google_content.get("parts") or []

        for part in parts:
            if isinstance(part, dict):
                # Text content
                if part.get("text"):
                    content = str(part["text"])

                # Function call from assistant
                elif part.get("function_call"):
                    function_call = part["function_call"]
                    if function_call and isinstance(function_call, dict):
                        function = FunctionSchema(
                            name=function_call.get("name", ""),
                            arguments=json.dumps(function_call.get("args", {})),
                        )
                        # Generate a unique ID for the tool call
                        tool_id = f"call_{time.time_ns()}"
                        tool = ToolSchema(
                            id=tool_id, function=function, thought_signature=part.get("thought_signature")
                        )
                        tools.append(tool)

                # Function response (tool result)
                elif part.get("function_response"):
                    chibi_role = "tool"
                    function_response = part["function_response"]
                    if function_response and isinstance(function_response, dict):
                        tool_call_id = function_response.get("name")
                        response_content = function_response.get("response", {})
                        if isinstance(response_content, dict):
                            content = response_content.get("content", "")

        return cls(
            role=chibi_role, content=content, tool_calls=tools or None, tool_call_id=tool_call_id, source="google"
        )

    def to_mistral(self) -> MistralSystemMessage | MistralUserMessage | MistralAssistantMessage | MistralToolMessage:
        """Convert to MistralAI SDK format."""
        if self.role == "system":
            return MistralSystemMessage(content=self.content, role="system")

        elif self.role == "user":
            return MistralUserMessage(content=self.content, role="user")

        elif self.role == "assistant":
            mistral_tool_calls: list[MistralToolCall] | None = None
            if self.tool_calls:
                mistral_tool_calls = [
                    MistralToolCall(
                        id=tool.id,
                        function=MistralFunctionCall(
                            name=tool.function.name,
                            arguments=tool.function.arguments or "{}",
                        ),
                    )
                    for tool in self.tool_calls
                ]
            return MistralAssistantMessage(
                content=self.content,
                tool_calls=mistral_tool_calls,
                role="assistant",
            )

        elif self.role == "tool":
            return MistralToolMessage(
                content=self.content,
                tool_call_id=self.tool_call_id or "",
                name=None,
                role="tool",
            )

        # Fallback to user message
        return MistralUserMessage(content=self.content, role="user")

    @classmethod
    def from_mistral(
        cls,
        mistral_message: MistralUserMessage | MistralAssistantMessage | MistralToolMessage,
    ) -> "Message":
        """Convert from MistralAI SDK format to Chibi Message."""
        role: Literal["user", "assistant", "tool"] = mistral_message.role  # type: ignore

        # Extract content - handle different content types
        raw_content = mistral_message.content
        if isinstance(raw_content, str):
            content = raw_content
        elif raw_content is None:
            content = ""
        else:
            # Content is a list/complex type - convert to string
            content = str(raw_content)

        tool_calls: list[ToolSchema] | None = None
        tool_call_id: str | None = None

        if isinstance(mistral_message, MistralAssistantMessage) and mistral_message.tool_calls:
            tool_calls = [
                ToolSchema(
                    id=tool.id or "",
                    function=FunctionSchema(
                        name=tool.function.name,
                        arguments=tool.function.arguments,
                    ),
                )
                for tool in mistral_message.tool_calls
            ]

        if isinstance(mistral_message, MistralToolMessage):
            # Handle Unset/None/str types
            raw_tool_call_id = mistral_message.tool_call_id
            if raw_tool_call_id and not isinstance(raw_tool_call_id, type(None)):
                tool_call_id = str(raw_tool_call_id)

        return cls(
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            source="mistral",
        )


class ImageMeta(BaseModel):
    id: int = Field(default_factory=time.time_ns)
    expire_at: float


class User(BaseModel):
    id: int
    alibaba_token: str | None = gpt_settings.alibaba_key
    anthropic_token: str | None = gpt_settings.anthropic_key
    deepseek_token: str | None = gpt_settings.deepseek_key
    gemini_token: str | None = gpt_settings.gemini_key
    mistralai_token: str | None = gpt_settings.mistralai_key
    openai_token: str | None = gpt_settings.openai_key
    tokens: dict[str, str] = {}
    messages: list[Message] = Field(default_factory=list)
    images: list[ImageMeta] = Field(default_factory=list)
    gpt_model: str | None = None  # Deprecated
    selected_gpt_model_name: str | None = None
    selected_gpt_provider_name: str | None = None
    selected_image_model_name: str | None = None
    selected_image_provider_name: str | None = None
    info: str = "No info provided"
    working_dir: str = application_settings.working_dir
    llm_skills: dict[str, str] = {}

    def __init__(self, **kwargs: Any) -> None:
        if kwargs.get("gpt_model", None) and not kwargs.get("selected_gpt_model_name", None):
            kwargs["selected_gpt_model_name"] = kwargs["gpt_model"]
        super().__init__(**kwargs)

    @property
    def providers(self) -> "RegisteredProviders":
        from chibi.services.providers import RegisteredProviders

        return RegisteredProviders(user_api_keys=self.tokens)

    @property
    def active_image_provider(self) -> "Provider":
        if not self.selected_image_provider_name:
            image_provider = self.providers.first_image_generation_ready
        else:
            image_provider = self.providers.get(provider_name=self.selected_image_provider_name)

        if not image_provider:
            raise NoApiKeyProvidedError(provider="Unset", detail="No API key provided")

        if not self.selected_image_provider_name:
            self.selected_image_provider_name = image_provider.name
            self.selected_image_model_name = image_provider.default_image_model
        return image_provider

    @property
    def stt_provider(self) -> "Provider":
        if gpt_settings.stt_provider:
            if provider := self.providers.get(gpt_settings.stt_provider):
                return provider
        if provider := self.providers.first_stt_ready:
            return provider
        raise ValueError("No stt-provider found.")

    @property
    def tts_provider(self) -> "Provider":
        if gpt_settings.tts_provider:
            if provider := self.providers.get(gpt_settings.tts_provider):
                return provider
        if provider := self.providers.first_tts_ready:
            return provider
        raise ValueError("No tts-provider found.")

    @property
    def moderation_provider(self) -> "Provider":
        if gpt_settings.moderation_provider:
            if provider := self.providers.get(gpt_settings.moderation_provider):
                return provider
        if provider := self.providers.first_moderation_ready:
            return provider
        raise ValueError("No moderation-provider found.")

    @property
    def active_gpt_provider(self) -> "Provider":
        if self.selected_gpt_provider_name:
            if provider := self.providers.get(provider_name=self.selected_gpt_provider_name):
                return provider

        if gpt_settings.default_provider:
            if provider := self.providers.get(provider_name=gpt_settings.default_provider):
                self.selected_gpt_provider_name = provider.name
                self.selected_gpt_model_name = gpt_settings.default_model or provider.default_model
                return provider

        if provider := self.providers.first_chat_ready:
            self.selected_gpt_provider_name = provider.name
            self.selected_gpt_model_name = provider.default_model
            return provider

        raise NoProviderSelectedError

    async def get_available_models(self, image_generation: bool = False) -> list[ModelChangeSchema]:
        providers = self.providers.available_instances
        tasks = [provider.get_available_models(image_generation=image_generation) for provider in providers]
        results = await asyncio.gather(*tasks)

        return list(itertools.chain.from_iterable(results))

    @property
    def has_reached_image_limits(self) -> bool:
        if not gpt_settings.image_generations_monthly_limit:
            return False
        if str(self.id) in gpt_settings.image_generations_whitelist:
            return False
        return len(self.images) >= gpt_settings.image_generations_monthly_limit
