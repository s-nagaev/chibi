import base64
from io import BytesIO

from openai import NOT_GIVEN

from chibi.config import gpt_settings
from chibi.services.providers.provider import (
    OpenAIFriendlyProvider,
    RestApiFriendlyProvider,
)


class Gemini(OpenAIFriendlyProvider, RestApiFriendlyProvider):
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    name = "Gemini"
    model_name_keywords = ["gemini"]
    model_name_keywords_exclude = ["image", "vision"]
    frequency_penalty = NOT_GIVEN
    image_quality = NOT_GIVEN
    image_size = NOT_GIVEN
    default_model = "models/gemini-2.0-flash"
    default_image_model = "models/imagen-3.0-generate-002"

    async def get_images(self, prompt: str, model: str | None = None) -> list[BytesIO]:
        base_url = "https://generativelanguage.googleapis.com/v1beta/"
        model = model or self.default_image_model
        params = {"key": self.token}

        if "image-gen" in model:
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseModalities": ["Text", "Image"]},
            }
            url = f"{base_url}{model}:generateContent"

        else:
            payload = {
                "instances": [{"prompt": prompt}],
                "parameters": {
                    "sampleCount": gpt_settings.image_n_choices,
                    "aspectRatio": gpt_settings.image_aspect_ratio,
                },
            }
            url = f"{base_url}{model}:predict"

        response = await self._request(method="POST", url=url, data=payload, params=params)
        response_data = response.json()
        if "image-gen" in model:
            image_data_b64 = response_data["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
            return [BytesIO(base64.b64decode(image_data_b64))]

        images = [x["bytesBase64Encoded"] for x in response_data["predictions"]]
        return [BytesIO(base64.b64decode(image_data_b64)) for image_data_b64 in images]

    @classmethod
    def is_image_ready_model(cls, model_name: str) -> bool:
        return "image" in model_name

    @property
    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json"}
