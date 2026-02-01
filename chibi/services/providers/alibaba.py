from http import HTTPStatus

import dashscope
from dashscope.aigc.image_generation import AioImageGeneration
from dashscope.api_entities.dashscope_response import Choice, ImageGenerationResponse, Message

from chibi.config import gpt_settings
from chibi.exceptions import ServiceResponseError
from chibi.schemas.app import ModelChangeSchema
from chibi.services.providers.provider import OpenAIFriendlyProvider

dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"


class Alibaba(OpenAIFriendlyProvider):
    api_key = gpt_settings.alibaba_key
    chat_ready = True
    image_generation_ready = True
    moderation_ready = True

    base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    name = "Alibaba"
    model_name_keywords = ["qwen"]
    model_name_keywords_exclude = ["tts", "stt"]
    default_model = "qwen-plus"
    default_moderation_model = "qwen-plus"
    max_tokens: int = 8192
    default_image_model = "qwen-image-plus"

    @staticmethod
    def _get_image_url(choice: Choice) -> str | None:
        if isinstance(choice.message.content, str):
            return choice.message.content
        return choice.message.content[0].get("image")

    async def get_images(self, prompt: str, model: str | None = None) -> list[str]:
        model = model or self.default_model
        message = Message(role="user", content=[{"text": prompt}])
        number_of_images = 1 if "qwen" in model or "z-image" in model else gpt_settings.image_n_choices
        response: ImageGenerationResponse = await AioImageGeneration.call(
            api_key=self.token,
            model=model,
            messages=[message],
            n=number_of_images,
            size=gpt_settings.image_size_alibaba,
            prompt_extend=True,
            watermark=False,
        )

        if response.status_code != HTTPStatus.OK:
            raise ServiceResponseError(
                provider=self.name,
                model=model,
                detail=(
                    f"Unexpected response status code: {response.status_code}. "
                    f"Response code: {response.code}. Message: {response.message}"
                ),
            )
        image_urls: list[str] = []
        for choice in response.output.choices:
            if url := self._get_image_url(choice):
                image_urls.append(url)
        return image_urls

    async def get_available_models(self, image_generation: bool = False) -> list[ModelChangeSchema]:
        models = await super().get_available_models(image_generation=image_generation)

        if image_generation:
            wan_models = [
                ModelChangeSchema(
                    provider=self.name,
                    name="wan2.6-t2i",
                    display_name="Wan 2.6",
                    image_generation=True,
                ),
            ]

            models += wan_models
        return models
