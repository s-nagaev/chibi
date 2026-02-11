from openai import NOT_GIVEN, Omit, omit
from openai.types import ImagesResponse, ReasoningEffort

from chibi.config import gpt_settings
from chibi.services.providers.provider import OpenAIFriendlyProvider


class Grok(OpenAIFriendlyProvider):
    api_key = gpt_settings.grok_key
    chat_ready = True
    image_generation_ready = True
    moderation_ready = True

    base_url = "https://api.x.ai/v1"
    name = "Grok"
    model_name_keywords = ["grok"]
    model_name_keywords_exclude = ["vision", "imag"]
    image_quality = NOT_GIVEN
    image_size = NOT_GIVEN
    default_image_model = "grok-2-image-1212"
    default_model = "grok-4-1-fast-reasoning"
    default_moderation_model = "grok-4-1-fast-non-reasoning"
    presence_penalty = NOT_GIVEN
    frequency_penalty = omit
    image_n_choices = 1

    async def _get_image_generation_response(self, prompt: str, model: str) -> ImagesResponse:
        return await self.client.images.generate(  # type: ignore
            model=model,
            prompt=prompt,
            n=gpt_settings.image_n_choices,
            quality=self.image_quality,
            size=self.image_size,
            timeout=gpt_settings.timeout,
            response_format="url",
            extra_body={"aspect_ratio": "16:9"},
        )

    def get_reasoning_effort_value(self, model_name: str) -> ReasoningEffort | Omit | None:
        if "grok-3-mini" in model_name:
            return "high"
        return omit
