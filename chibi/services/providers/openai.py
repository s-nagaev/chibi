from openai import NOT_GIVEN
from openai.types import ImagesResponse
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
)

from chibi.config import gpt_settings
from chibi.schemas.app import ChatResponseSchema
from chibi.services.providers.provider import OpenAIFriendlyProvider


class OpenAI(OpenAIFriendlyProvider):
    name = "OpenAI"
    model_name_prefixes = ["gpt", "o1", "o3", "o4"]
    model_name_keywords_exclude = ["audio", "realtime", "transcribe"]
    base_url = "https://api.openai.com/v1"
    max_tokens = NOT_GIVEN
    default_model = "gpt-4.1"
    default_image_model = "dall-e-3"

    async def get_chat_response(
        self,
        messages: list[ChatCompletionMessageParam],
        model: str | None = None,
        system_prompt: str = gpt_settings.assistant_prompt,
    ) -> ChatResponseSchema:
        model = model or self.default_model

        self.presence_penalty = self.presence_penalty if "search" not in model else NOT_GIVEN
        self.temperature = self.temperature if "search" not in model else NOT_GIVEN
        self.frequency_penalty = self.frequency_penalty if "search" not in model else NOT_GIVEN

        if model.lower().startswith("o1-preview") or model.lower().startswith("o1-mini"):
            dialog: list[ChatCompletionMessageParam] = messages
        else:
            system_message = ChatCompletionSystemMessageParam(role="system", content=system_prompt)
            dialog = [system_message] + messages

        return await self._get_chat_completion_response(messages=dialog, model=model, system_prompt=system_prompt)

    @classmethod
    def is_image_ready_model(cls, model_name: str) -> bool:
        return "dall-e" in model_name

    async def _get_image_generation_response(self, prompt: str, model: str) -> ImagesResponse:
        return await self.client.images.generate(
            model=model,
            prompt=prompt,
            n=gpt_settings.image_n_choices if "dall-e-2" in model else 1,
            quality=self.image_quality,
            size=self.image_size if "dall-e-3" in model else NOT_GIVEN,
            timeout=gpt_settings.timeout,
            response_format="url",
        )
