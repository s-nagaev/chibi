from typing import Iterable

from openai import NOT_GIVEN
from openai.types import ImagesResponse
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion import Choice

from chibi.config import gpt_settings
from chibi.schemas.app import ChatCompletionMessageSchema, ChatResponseSchema
from chibi.services.providers.provider import OpenAIFriendlyProvider


class OpenAI(OpenAIFriendlyProvider):
    name = "OpenAI"
    model_name_prefixes = ["gpt", "o1", "o3"]
    model_name_keywords_exclude = ["audio", "realtime", "transcribe"]
    base_url = "https://api.openai.com/v1"
    max_tokens = NOT_GIVEN
    default_model = "o3-mini"
    default_image_model = "dall-e-3"

    async def _get_chat_completion_response(
        self,
        messages: list[ChatCompletionMessageSchema],
        model: str,
        system_prompt: str,
    ) -> ChatResponseSchema:
        if model.lower().startswith("o1-preview") or model.lower().startswith("o1-mini"):
            dialog: Iterable[ChatCompletionMessageParam] = messages  # type: ignore
        else:
            system_message = ChatCompletionSystemMessageParam(role="system", content=system_prompt)
            dialog: Iterable[ChatCompletionMessageParam] = [system_message] + messages  # type: ignore

        response = await self.client.chat.completions.create(
            model=model,
            messages=dialog,
            max_completion_tokens=self.max_tokens,
            max_tokens=self.max_tokens,
            presence_penalty=self.presence_penalty if "search" not in model else NOT_GIVEN,
            temperature=self.temperature if "search" not in model else NOT_GIVEN,
            frequency_penalty=self.frequency_penalty if "search" not in model else NOT_GIVEN,
            timeout=self.timeout,
        )
        if len(response.choices) != 0:
            choices: list[Choice] = response.choices
            data = choices[0]
            answer = data.message.content
            usage = response.usage
        else:
            answer = ""
            usage = None

        return ChatResponseSchema(answer=answer or "", provider=self.name, model=model, usage=usage)

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
