import base64
import json
from io import BytesIO
from typing import Any

from loguru import logger
from openai import NOT_GIVEN, BadRequestError, NotFoundError, Omit, omit
from openai.types import ImagesResponse, ReasoningEffort
from openai.types.chat import ChatCompletionContentPartTextParam, ChatCompletionUserMessageParam
from openai.types.chat.chat_completion_content_part_param import File, FileFile
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion

from chibi.config import application_settings, gpt_settings
from chibi.exceptions import NoModelSelectedError, ServiceResponseError
from chibi.models import FunctionSchema, Message, ToolSchema, User
from chibi.schemas.app import ChatResponseSchema, UsageSchema, VisionResultSchema
from chibi.services.interface import UserInterface
from chibi.services.metrics import MetricsService
from chibi.services.providers.provider import OpenAIFriendlyProvider
from chibi.services.providers.tools import RegisteredChibiTools
from chibi.services.providers.tools.schemas import ToolCallSchema
from chibi.services.providers.utils import get_usage_msg, prepare_system_prompt, send_llm_thoughts

RESPONSES_ONLY_MODELS = {"gpt-5.2", "gpt-5.3", "gpt-5.4", "codex"}


class OpenAI(OpenAIFriendlyProvider):
    api_key = gpt_settings.openai_key
    chat_ready = True
    tts_ready = True
    stt_ready = True
    image_generation_ready = True
    moderation_ready = True
    vision_ready = True
    ocr_ready = True

    name = "OpenAI"
    model_name_prefixes = ["gpt", "o1", "o3", "o4"]
    model_name_keywords_exclude = ["audio", "realtime", "transcribe", "tts", "image"]
    base_url = "https://api.openai.com/v1"
    max_tokens = NOT_GIVEN
    default_model = "gpt-5.2"
    default_image_model = "gpt-image-2"
    default_moderation_model = "gpt-5-mini"
    default_stt_model = "gpt-4o-transcribe"
    default_tts_model = "gpt-4o-mini-tts"
    default_tts_voice = "nova"
    default_vision_model = "gpt-5-mini"
    default_ocr_model = "gpt-5-mini"

    async def transcribe(self, audio: BytesIO, model: str | None = None) -> str:
        model = model or self.default_stt_model
        logger.info(f"Transcribing audio with model {model}...")
        response = await self.client.audio.transcriptions.create(
            model=model,
            file=("voice.ogg", audio.getvalue()),
        )
        if response:
            logger.info(f"Transcribed text: {response.text}")
            return response.text
        raise ValueError("Could not transcribe audio message")

    async def get_chat_response(
        self,
        messages: list[Message],
        user: User,
        model: str | None = None,
        system_prompt: str = gpt_settings.assistant_prompt,
        interface: UserInterface | None = None,
    ) -> tuple[ChatResponseSchema, list[Message]]:
        """Get a chat response with Responses API fallback to Chat Completions.

        Args:
            messages: List of conversation messages.
            user: The user requesting the response.
            model: The model to use. Defaults to self.default_model.
            system_prompt: System prompt for the assistant.
            interface: Optional user interface for sending thoughts.

        Returns:
            A tuple of the chat response schema and updated messages list.

        Raises:
            BadRequestError: If model is responses-only and request is invalid.
            NotFoundError: If model is responses-only and not found.
        """
        model = model or self.default_model

        try:
            return await self._get_response_completion_response(
                messages=messages,
                model=model,
                user=user,
                system_prompt=system_prompt,
                interface=interface,
            )
        except (BadRequestError, NotFoundError) as e:
            # Only fallback for 400/404 errors (unsupported model or parameter)
            for substr in RESPONSES_ONLY_MODELS:
                if substr in model:
                    logger.warning(f"Responses-only model {model} failed: {e}. Not falling back.")
                    raise

            logger.warning(f"Responses API failed for {model}: {e}. Falling back to Chat Completions.")
            return await super().get_chat_response(
                messages=messages,
                user=user,
                model=model,
                system_prompt=system_prompt,
                interface=interface,
            )

    async def speech(self, text: str, voice: str | None = None, model: str | None = None) -> bytes:
        voice = voice or self.tts_voice
        model = model or self.tts_model
        logger.info(f"Recording a voice message with model {model}...")
        response = await self.client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
        )
        return await response.aread()

    @classmethod
    def is_image_ready_model(cls, model_name: str) -> bool:
        return "image" in model_name.lower()

    async def _get_image_generation_response(self, prompt: str, model: str) -> ImagesResponse:
        return await self.client.images.generate(
            model=model,
            prompt=prompt,
            n=gpt_settings.image_n_choices,
            quality=self.image_quality,
            size=self.image_size,
            timeout=gpt_settings.timeout,
        )

    def get_model_display_name(self, model_name: str) -> str:
        if "dall" in model_name:
            return model_name.replace("dall-e-", "DALL·E ")

        model_display_name = super().get_model_display_name(model_name=model_name)

        if "Gpt" in model_display_name:
            model_display_name = model_display_name.replace("Gpt ", "GPT-")
        return model_display_name

    def get_reasoning_effort_value(self, model_name: str) -> ReasoningEffort | Omit | None:
        if "chat" in model_name:
            return omit
        if "gpt-5" in model_name:
            return "medium"
        return omit

    def _get_temperature_value(self, model_name: str) -> float | Omit:
        if model_name.startswith("o"):
            return omit
        if model_name.startswith("gpt-5"):
            return omit
        return getattr(self, "temperature", gpt_settings.temperature)

    async def _get_response_completion_response(
        self,
        messages: list[Message],
        model: str,
        user: User,
        system_prompt: str | None = None,
        interface: UserInterface | None = None,
    ) -> tuple[ChatResponseSchema, list[Message]]:
        """Get a chat response using the OpenAI Responses API.

        Args:
            messages: List of conversation messages.
            model: The model to use for the response.
            user: The user requesting the response.
            system_prompt: Optional system prompt to use as instructions.
            interface: Optional user interface for sending thoughts.

        Returns:
            A tuple of the chat response schema and updated messages list.

        Raises:
            ServiceResponseError: If the response contains no output.
        """
        input_items: list[dict] = []
        for msg in messages:
            input_items.extend(msg.to_responses_items())

        instructions: str | Omit = omit
        if system_prompt:
            instructions = await prepare_system_prompt(
                base_system_prompt=system_prompt, user_id=user.id, interface=interface
            )

        reasoning_effort = self.get_reasoning_effort_value(model_name=model)
        tools = [self.adapt_tool_for_responses(t) for t in RegisteredChibiTools.get_tool_definitions()]

        request_kwargs: dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": input_items,
            "tools": tools,
            "tool_choice": "auto",
            "max_output_tokens": self._get_max_tokens_value(model_name=model),
            "temperature": self._get_temperature_value(model_name=model),
            "timeout": self.timeout,
            "stream": False,
        }

        if reasoning_effort not in (omit, None):
            request_kwargs["reasoning"] = {"effort": reasoning_effort}

        response = await self.get_client().responses.create(**request_kwargs)

        if not response.output:
            raise ServiceResponseError(
                provider=self.name,
                model=model,
                detail="Unexpected (empty) response received",
            )

        answer: str = response.output_text or ""

        # Extract reasoning from response.output if present
        reasoning_items = [item for item in response.output if getattr(item, "type", None) == "reasoning"]
        if reasoning_items:
            reasoning_text = " ".join(
                getattr(item, "summary", "") or getattr(item, "text", "") or "" for item in reasoning_items
            ).strip()
            if reasoning_text:
                await send_llm_thoughts(thoughts=reasoning_text, interface=interface)

        usage = UsageSchema()
        if response.usage:
            usage = UsageSchema(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.total_tokens,
            )

        if application_settings.is_influx_configured:
            MetricsService.send_usage_metrics(metric=usage, model=model, provider=self.name, user=user)
        usage_message = get_usage_msg(usage=usage)

        tool_call_items = [item for item in response.output if getattr(item, "type", None) == "function_call"]

        if not tool_call_items:
            messages.append(Message(role="assistant", content=answer))
            return ChatResponseSchema(answer=answer, provider=self.name, model=model, usage=usage), messages

        logger.log("CALL", f"{model} requested the call of {len(tool_call_items)} tools.")

        thoughts = answer or "No thoughts"
        if answer:
            await send_llm_thoughts(thoughts=thoughts, interface=interface)
        logger.log("THINK", f"{model}: {thoughts}. {usage_message}")

        calls = [
            ToolCallSchema(
                tool_name=item.name,
                args=json.loads(item.arguments) if item.arguments else {},
            )
            for item in tool_call_items
        ]
        results = await self.call_functions(
            calls=calls, caller_model=model, caller_provider=self.name, user_id=user.id, interface=interface
        )

        assistant_message = Message(
            role="assistant",
            content=answer,
            tool_calls=[
                ToolSchema(
                    id=item.call_id,
                    type="function",
                    function=FunctionSchema(
                        name=item.name,
                        arguments=item.arguments,
                    ),
                )
                for item in tool_call_items
            ],
        )
        messages.append(assistant_message)

        for item, result in zip(tool_call_items, results):
            tool_result_message = Message(
                role="tool",
                content=result.model_dump_json(),
                tool_call_id=item.call_id,
            )
            messages.append(tool_result_message)

        logger.log("CALL", "All the function results have been obtained. Returning them to the LLM...")
        return await self._get_response_completion_response(
            messages=messages, model=model, user=user, system_prompt=system_prompt, interface=interface
        )

    async def ocr(self, pdf: bytes, model: str | None = None) -> VisionResultSchema:
        """Extract text from a PDF document using OpenAI's vision/OCR capabilities.

        Args:
            pdf: The PDF file content as bytes.
            model: The model to use for OCR. Defaults to default_vision_model.

        Returns:
            VisionResultSchema containing the extracted text and descriptions.
        """
        model = model or self.default_ocr_model
        logger.info(f"[{self.name}] Extracting text from PDF with model {model}...")

        # Encode PDF to base64
        pdf_base64 = base64.b64encode(pdf).decode("utf-8")
        file_data = f"data:application/pdf;base64,{pdf_base64}"

        # Build the message content with the PDF file
        content: list[ChatCompletionContentPartTextParam | File] = [
            File(
                type="file",
                file=FileFile(
                    filename="document.pdf",
                    file_data=file_data,
                ),
            ),
            ChatCompletionContentPartTextParam(
                type="text",
                text="Extract all text from this PDF. Provide a short description and full text content.",
            ),
        ]

        # Use parse() for structured output with Pydantic models
        response: ParsedChatCompletion[VisionResultSchema] = await self.get_client().chat.completions.parse(
            model=model,
            messages=[
                ChatCompletionUserMessageParam(
                    role="user",
                    content=content,
                )
            ],
            response_format=VisionResultSchema,
            # max_tokens=4096,
        )

        if not response.choices:
            raise ServiceResponseError(
                provider=self.name,
                model=model,
                detail="Could not extract text from PDF: empty response",
            )

        result = response.choices[0].message
        if not result or not result.parsed:
            raise ServiceResponseError(
                provider=self.name,
                model=model,
                detail="Could not extract text from PDF: empty parsed result",
            )

        logger.info(f"[{self.name}] PDF text extracted successfully: {result.parsed.short_description}...")
        return result.parsed

    async def vision(
        self,
        image: bytes,
        mime_type: str,
        model: str | None = None,
        prompt: str | None = None,
    ) -> VisionResultSchema:
        """Analyze an image using OpenAI's Responses API with structured output.

        Args:
            image: The image file content as bytes.
            mime_type: The MIME type of the image (e.g., "image/jpeg").
            model: The model to use for vision analysis. Defaults to default_vision_model.
            prompt: Custom prompt for the vision analysis. Defaults to a generic description prompt.

        Returns:
            VisionResultSchema containing the image description and extracted text.

        Raises:
            NoModelSelectedError: If no vision model is configured.
            ServiceResponseError: If the response is empty or cannot be parsed.
        """
        model = model or self.default_vision_model
        if not model:
            raise NoModelSelectedError(provider=self.name, detail="No vision model selected")
        prompt = prompt or "Describe the image in detail."
        logger.info(f"[{self.name}] Analyzing image with model {model}...")

        image_base64 = base64.b64encode(image).decode("utf-8")
        data_url = f"data:{mime_type};base64,{image_base64}"

        request_kwargs: dict[str, Any] = {
            "model": model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "vision_result",
                    "schema": VisionResultSchema.model_json_schema(),
                }
            },
            "max_output_tokens": 4096,
        }

        response = await self.get_client().responses.create(**request_kwargs)

        if not response.output_text:
            raise ServiceResponseError(
                provider=self.name,
                model=model,
                detail="Could not analyze image: empty response",
            )

        try:
            result = VisionResultSchema.model_validate_json(response.output_text)
        except Exception as e:
            raise ServiceResponseError(
                provider=self.name,
                model=model,
                detail=f"Could not parse vision response: {e}",
            )

        logger.info(f"[{self.name}] Image analyzed successfully: {result.short_description}...")
        return result
