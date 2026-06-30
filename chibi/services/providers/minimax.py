import base64

from anthropic import AsyncClient
from loguru import logger

from chibi.config import gpt_settings
from chibi.exceptions import NoApiKeyProvidedError
from chibi.schemas.app import ModelChangeSchema
from chibi.services.providers.provider import AnthropicFriendlyProvider


class Minimax(AnthropicFriendlyProvider):
    api_key = gpt_settings.minimax_api_key
    chat_ready = True
    tts_ready = True
    moderation_ready = True
    image_to_image_ready = True

    name = "Minimax"
    base_url = "https://api.minimax.io/anthropic"
    default_model = "MiniMax-M2.7"
    default_moderation_model = "MiniMax-M2.5-lighting"
    model_name_keywords = ["MiniMax"]

    base_tts_url = "https://api.minimax.io/v1/"
    default_tts_model = "speech-2.8-turbo"
    default_tts_voice = "Korean_HaughtyLady"
    default_image_model = "image-01"
    default_image_to_image_model = "image-01"

    def __init__(self, token: str) -> None:
        self._client: AsyncClient | None = None
        super().__init__(token=token)

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }

    @property
    def client(self) -> AsyncClient:
        if self._client:
            return self._client

        if not self.token:
            raise NoApiKeyProvidedError(provider=self.name)

        self._client = AsyncClient(api_key=self.token, base_url=self.base_url)
        return self._client

    async def get_available_models(
        self, image_generation: bool = False, image_to_image: bool = False
    ) -> list[ModelChangeSchema]:
        if image_to_image:
            # MiniMax i2i supports only by image-01.
            i2i_models = [
                ModelChangeSchema(
                    provider=self.name,
                    name="image-01",
                    display_name="Image-01 (i2i)",
                    image_generation=False,
                    image_to_image=True,
                )
            ]
            return self.filter_and_return_list_of_models(models=i2i_models, image_to_image=image_to_image)

        if image_generation:
            image_models = [
                ModelChangeSchema(provider=self.name, name="image-01", display_name="Image-01", image_generation=True)
            ]

            return self.filter_and_return_list_of_models(models=image_models, image_generation=image_generation)

        models = await super().get_available_models()
        if not models:
            # Get models endpoint sometimes returns empty list, so we need a hacky fallback here
            supported_models = [
                "MiniMax-M3",
                "MiniMax-M2.7",
                "MiniMax-M2.7-highspeed",
                "MiniMax-M2.5",
                "MiniMax-M2.5-highspeed",
            ]
            models = [
                ModelChangeSchema(
                    provider=self.name,
                    name=model_name,
                    display_name=model_name,
                    image_generation=False,
                )
                for model_name in supported_models
            ]
        return self.filter_and_return_list_of_models(models=models, image_generation=image_generation)

    async def image_to_image(
        self,
        prompt: str,
        input_image: bytes,
        mime_type: str,
        model: str | None = None,
    ) -> list[str]:
        selected_model = model or self.default_image_to_image_model
        image_b64 = base64.b64encode(input_image).decode("ascii")
        data_uri = f"data:{mime_type};base64,{image_b64}"

        url = "https://api.minimax.io/v1/image_generation"
        logger.info(f"Generating image-to-image with model {selected_model}...")

        response = await self._request(
            method="POST",
            url=url,
            data={
                "model": selected_model,
                "prompt": prompt,
                "aspect_ratio": gpt_settings.image_aspect_ratio,
                "response_format": "url",
                "n": gpt_settings.image_n_choices,
                "prompt_optimizer": True,
                "reference_image": data_uri,
            },
        )
        response_data = response.json()
        images_urls = response_data.get("data", {}).get("image_urls", [])
        return images_urls

    async def speech(self, text: str, voice: str | None = None, model: str | None = None) -> bytes:
        voice = voice or self.tts_voice
        model = model or self.tts_model

        logger.info(f"Recording a voice message with model {model}...")

        url = f"{self.base_tts_url}t2a_v2"

        data = {
            "model": model,
            "text": text,
            "voice_setting": {
                "voice_id": voice,
                "emotion": "happy",
                "speed": 1.2,
            },
        }
        try:
            response = await self._request(method="POST", url=url, data=data)
        except Exception as e:
            logger.error(f"Failed to get available models for provider {self.name} due to exception: {e}")
            return bytes()
        response_data = response.json()["data"]
        return bytes.fromhex(response_data["audio"])

    async def get_images(self, prompt: str, model: str | None = None) -> list[str]:
        url = "https://api.minimax.io/v1/image_generation"
        response = await self._request(
            method="POST",
            url=url,
            data={
                "model": model,
                "prompt": prompt,
                "aspect_ratio": "16:9",
                "response_format": "url",
                "n": gpt_settings.image_n_choices,
                "prompt_optimizer": True,
            },
        )
        response_data = response.json()
        images_urls = response_data.get("data", {}).get("image_urls", [])
        return images_urls
