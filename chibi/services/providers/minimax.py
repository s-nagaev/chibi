from loguru import logger

from chibi.config import gpt_settings
from chibi.schemas.app import ModelChangeSchema
from chibi.services.providers.provider import RestApiFriendlyProvider


class Minimax(RestApiFriendlyProvider):
    api_key = gpt_settings.minimax_api_key
    chat_ready = False
    tts_ready = True

    name = "Minimax"

    base_url = "https://api.minimax.io/v1/"
    default_tts_model = "speech-2.6-hd"
    default_tts_voice = "English_WhimsicalGirl"

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }

    async def speech(
        self, text: str, voice: str | None = default_tts_voice, model: str | None = default_tts_model
    ) -> bytes:
        logger.info(f"Recording a voice message with model {model}...")

        url = f"{self.base_url}t2a_v2"

        data = {
            "model": model,
            "text": text,
            "voice_setting": {
                "voice_id": voice,
            },
        }
        try:
            response = await self._request(method="POST", url=url, data=data)
        except Exception as e:
            logger.error(f"Failed to get available models for provider {self.name} due to exception: {e}")
            return bytes()
        response_data = response.json()["data"]
        return bytes.fromhex(response_data["audio"])

    async def get_available_models(self, image_generation: bool = False) -> list[ModelChangeSchema]:
        return []
