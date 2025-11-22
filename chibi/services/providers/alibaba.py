from http import HTTPStatus

import dashscope
from dashscope.aigc import AioImageSynthesis
from dashscope.api_entities.dashscope_response import ImageSynthesisResponse

from chibi.config import gpt_settings
from chibi.exceptions import ServiceResponseError
from chibi.schemas.app import ModelChangeSchema
from chibi.services.providers.provider import OpenAIFriendlyProvider

dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"


class Alibaba(OpenAIFriendlyProvider):
    api_key = gpt_settings.alibaba_key
    chat_ready = True
    image_generation_ready = True

    base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    name = "Alibaba"
    model_name_keywords = ["qwen"]
    model_name_keywords_exclude = ["tts", "stt"]
    default_model = "qwen-plus"
    max_tokens: int = 8192
    default_image_model = "qwen-image-plus"

    async def get_images(self, prompt: str, model: str | None = None) -> list[str]:
        response: ImageSynthesisResponse = await AioImageSynthesis.call(
            api_key=self.token,
            model=model or self.default_model,
            prompt=prompt,
            n=1,
            size="1920*1080" if model == "wan2.5-t2i-preview" else "1664*928",
            prompt_extend=True,
            watermark=False,
        )

        if response.status_code != HTTPStatus.OK:
            raise ServiceResponseError(
                provider=self.name,
                model=model or self.default_model,
                detail=(
                    f"Unexpected response status code: {response.status_code}. "
                    f"Response code: {response.code}. Message: {response.message}"
                ),
            )
        return [result.url for result in response.output.results]

    async def get_available_models(self, image_generation: bool = False) -> list[ModelChangeSchema]:
        models = await super().get_available_models(image_generation=image_generation)

        if image_generation:
            wan_models = [
                ModelChangeSchema(
                    provider=self.name,
                    name="wan2.5-t2i-preview",
                    display_name="Wan 2.5 Preview",
                    image_generation=True,
                ),
                ModelChangeSchema(
                    provider=self.name,
                    name="wan2.2-t2i-flash",
                    display_name="Wan 2.2 Flash",
                    image_generation=True,
                ),
                ModelChangeSchema(
                    provider=self.name,
                    name="wan2.2-t2i-plus",
                    display_name="Wan 2.2 Plus",
                    image_generation=True,
                ),
            ]

            models += wan_models
        return models
