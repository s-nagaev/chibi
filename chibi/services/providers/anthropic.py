import base64
from typing import cast

from anthropic import AsyncClient
from anthropic.types import Base64ImageSourceParam, ImageBlockParam, TextBlock, TextBlockParam
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
        image_source: Base64ImageSourceParam = cast(
            Base64ImageSourceParam,
            {"type": "base64", "media_type": mime_type, "data": image_base64},
        )
        content: list[ImageBlockParam | TextBlockParam] = [
            cast(ImageBlockParam, {"type": "image", "source": image_source}),
            cast(
                TextBlockParam,
                {
                    "type": "text",
                    "text": (
                        f"{prompt}. Your response must be valid JSON with the following "
                        'structure: {"short_description": "<max 100 characters>", "full_description": '
                        '"<detailed description>", "text": "<extracted text if any, otherwise null>"}'
                    ),
                },
            ),
        ]
        response = await self.client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
        )

        if not response.content:
            raise ServiceResponseError(
                provider=self.name,
                model=model,
                detail=f"Could not analyze image: empty response: {response}",
            )

        first_content = response.content[0]
        if not isinstance(first_content, TextBlock):
            raise ServiceResponseError(
                provider=self.name,
                model=model,
                detail=f"Could not analyze image: unexpected content type: {type(first_content)}",
            )

        result_text = first_content.text

        if not result_text:
            raise ServiceResponseError(
                provider=self.name,
                model=model,
                detail=f"Could not analyze image: empty result text: {response}",
            )

        logger.info(f"Image analyzed successfully: {result_text[:50]}...")
        return VisionResultSchema.model_validate_json(result_text)
