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

    name = "Minimax"
    base_url = "https://api.minimax.io/anthropic"
    default_model = "MiniMax-M2.1"
    default_moderation_model = "MiniMax-M2.1-lighting"

    base_tts_url = "https://api.minimax.io/v1/"
    default_tts_model = "speech-2.8-turbo"
    default_tts_voice = "Korean_HaughtyLady"

    def __init__(self, token: str) -> None:
        self._client: AsyncClient | None = None
        super().__init__(token=token)

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "x-api-key": self.token,
            "anthropic-version": "2023-06-01",
        }

    @property
    def tts_headers(self) -> dict[str, str]:
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

    async def get_available_models(self, image_generation: bool = False) -> list[ModelChangeSchema]:
        supported_models = [
            "MiniMax-M2.1",
            "MiniMax-M2.1-lightning",
            "MiniMax-M2",
        ]
        return [
            ModelChangeSchema(
                provider=self.name,
                name=model_name,
                display_name=model_name,
                image_generation=False,
            )
            for model_name in supported_models
            if not gpt_settings.models_whitelist or model_name in gpt_settings.models_whitelist
        ]

    async def speech(
        self, text: str, voice: str | None = default_tts_voice, model: str | None = default_tts_model
    ) -> bytes:
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
            response = await self._request(method="POST", url=url, data=data, headers=self.tts_headers)
        except Exception as e:
            logger.error(f"Failed to get available models for provider {self.name} due to exception: {e}")
            return bytes()
        response_data = response.json()["data"]
        return bytes.fromhex(response_data["audio"])
