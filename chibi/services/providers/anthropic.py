import base64
from typing import Literal, cast

from anthropic import AsyncClient
from anthropic.types import (
    Base64ImageSourceParam,
    Base64PDFSourceParam,
    DocumentBlockParam,
    ImageBlockParam,
    MessageParam,
    TextBlockParam,
    ToolChoiceToolParam,
    ToolParam,
    ToolUseBlock,
)
from anthropic.types.tool_param import InputSchemaTyped
from loguru import logger

from chibi.config import gpt_settings
from chibi.exceptions import NoApiKeyProvidedError
from chibi.schemas.app import VisionResultSchema
from chibi.services.providers.provider import AnthropicFriendlyProvider, ServiceResponseError


class Anthropic(AnthropicFriendlyProvider):
    api_key = gpt_settings.anthropic_key
    chat_ready = True
    moderation_ready = True
    vision_ready = True
    ocr_ready = True

    name = "Anthropic"
    model_name_keywords = ["claude"]
    default_model = "claude-sonnet-4-5-20250929"
    default_moderation_model = "claude-haiku-4-5-20251001"
    default_vision_model = "claude-sonnet-4-5-20250929"

    def __init__(self, token: str) -> None:
        self._client: AsyncClient | None = None
        super().__init__(token=token)

    @property
    def client(self) -> AsyncClient:
        if self._client:
            return self._client

        if not self.token:
            raise NoApiKeyProvidedError(provider=self.name)

        self._client = AsyncClient(api_key=self.token)
        return self._client

    @client.setter
    def client(self, value: AsyncClient) -> None:
        """Setter for client property to allow mocking in tests."""
        self._client = value

    def get_client(self) -> AsyncClient:
        """Get the client, checking for mock first."""
        return self.client

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "x-api-key": self.token,
            "anthropic-version": "2023-06-01",
        }

    async def vision(
        self, image: bytes, mime_type: str, model: str | None = None, prompt: str | None = None
    ) -> VisionResultSchema:
        model = model or self.default_vision_model
        prompt = prompt or "Describe the image in detail"
        logger.info(f"Analyzing image with model {model}...")

        # Encode image to base64
        image_base64 = base64.b64encode(image).decode("utf-8")

        # Use the Anthropic messages API with vision
        # Use cast to satisfy mypy's strict literal type checking
        image_source = Base64ImageSourceParam(
            type="base64",
            media_type=cast(Literal["image/jpeg", "image/png", "image/gif", "image/webp"], mime_type),
            data=image_base64,
        )
        content: list[ImageBlockParam | TextBlockParam] = [
            ImageBlockParam(type="image", source=image_source),
            TextBlockParam(
                type="text",
                text=prompt,
            ),
        ]
        tool = ToolParam(
            name="analyze_and_describe_image",
            description="Analyze and describe the image",
            input_schema=InputSchemaTyped(
                type="object",
                properties={
                    "short_description": {
                        "type": "string",
                        "description": "Image short description, up to 100 characters",
                    },
                    "full_description": {
                        "type": "string",
                        "description": "Image full description",
                    },
                    "text": {
                        "type": "string",
                        "description": "Extracted text from the image",
                    },
                },
                required=["short_description", "full_description"],
            ),
        )
        response_message = await self.client.messages.create(
            model=model,
            max_tokens=16384,
            messages=[
                MessageParam(
                    role="user",
                    content=content,
                )
            ],
            timeout=self.timeout,
            temperature=0.1,
            tools=[tool],
            tool_choice=ToolChoiceToolParam(type="tool", name="analyze_and_describe_image"),
        )
        tool_call: ToolUseBlock | None = next(
            (part for part in response_message.content if isinstance(part, ToolUseBlock)), None
        )
        if not tool_call:
            raise ServiceResponseError(
                provider=self.name,
                model=model,
                detail=f"Could not analyze image: empty response: {response_message}",
            )

        answer = tool_call.input
        try:
            result = VisionResultSchema.model_validate(answer)
            logger.info(f"[{self.name}] Image analyzed successfully: {result.short_description}...")
            return result
        except Exception as e:
            raise ServiceResponseError(
                provider=self.name,
                model=model,
                detail=f"Could not analyze image. Error parsing result: {e}",
            )

    async def ocr(self, pdf: bytes, model: str | None = None) -> VisionResultSchema:
        """Extract text from a PDF document using Anthropic Claude's document vision.

        Args:
            pdf: The PDF file content as bytes.
            model: The model to use for OCR. Defaults to default_vision_model.

        Returns:
            VisionResultSchema containing the extracted text and descriptions.
        """
        model = model or self.default_vision_model
        logger.info(f"[{self.name}] Extracting text from PDF with model {model}...")

        # Encode PDF to base64
        pdf_base64 = base64.b64encode(pdf).decode("utf-8")

        # Build the document content block for Anthropic Messages API
        # Use cast to satisfy mypy's strict literal type checking
        document_source = Base64PDFSourceParam(
            type="base64",
            media_type="application/pdf",
            data=pdf_base64,
        )

        content: list[DocumentBlockParam | TextBlockParam] = [
            DocumentBlockParam(
                type="document",
                source=document_source,
            ),
            TextBlockParam(
                type="text",
                text="Extract all text from this PDF",
            ),
        ]

        tool = ToolParam(
            name="print_pdf_ocr_result",
            description="Extract text from a PDF document",
            input_schema=InputSchemaTyped(
                type="object",
                properties={
                    "short_description": {
                        "type": "string",
                        "description": "Document short description, up to 100 characters",
                    },
                    "full_description": {
                        "type": "string",
                        "description": "Document full description",
                    },
                    "text": {
                        "type": "string",
                        "description": "Extracted text from the PDF document",
                    },
                },
                required=["short_description", "text"],
            ),
        )
        response_message = await self.client.messages.create(
            model=model,
            max_tokens=16384,
            messages=[
                MessageParam(
                    role="user",
                    content=content,
                )
            ],
            timeout=self.timeout,
            temperature=0.1,
            tools=[tool],
            tool_choice=ToolChoiceToolParam(type="tool", name="print_pdf_ocr_result"),
        )
        tool_call: ToolUseBlock | None = next(
            (part for part in response_message.content if isinstance(part, ToolUseBlock)), None
        )
        if not tool_call:
            raise ServiceResponseError(
                provider=self.name,
                model=model,
                detail="Could not extract text from PDF: no tool use block found",
            )

        answer = tool_call.input
        try:
            result = VisionResultSchema.model_validate(answer)
            logger.info(f"[{self.name}] PDF text extracted successfully: {result.short_description}...")
            return result
        except Exception as e:
            raise ServiceResponseError(
                provider=self.name,
                model=model,
                detail=f"Could not extract text from PDF. Error parsing result: {e}",
            )
