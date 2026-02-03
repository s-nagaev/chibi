from io import BytesIO

from loguru import logger
from openai import NOT_GIVEN
from openai.types import ImagesResponse

from chibi.config import gpt_settings
from chibi.constants import OPENAI_TTS_INSTRUCTIONS
from chibi.services.providers.provider import OpenAIFriendlyProvider


class OpenAI(OpenAIFriendlyProvider):
    api_key = gpt_settings.openai_key
    chat_ready = True
    tts_ready = True
    stt_ready = True
    image_generation_ready = True
    moderation_ready = True

    name = "OpenAI"
    model_name_prefixes = ["gpt", "o1", "o3", "o4"]
    model_name_keywords_exclude = ["audio", "realtime", "transcribe", "tts", "image"]
    base_url = "https://api.openai.com/v1"
    max_tokens = NOT_GIVEN
    default_model = "gpt-5.2"
    default_image_model = "dall-e-3"
    default_moderation_model = "gpt-5-mini"
    default_stt_model = "whisper-1"
    default_tts_model = "gpt-4o-mini-tts"
    default_tts_voice = "nova"

    async def transcribe(self, audio: BytesIO, model: str | None = None) -> str:
        model = model or self.default_stt_model
        logger.info(f"Transcribing audio with model {model}...")
        response = await self.client.audio.transcriptions.create(
            model=model,
            file=("voice.ogg", audio.getvalue()),
        )
        if response:
            logger.info(f"Transcribed text: {response.text}")
            return response.text
        raise ValueError("Could not transcribe audio message")

    async def speech(self, text: str, voice: str | None = None, model: str | None = None) -> bytes:
        voice = voice or self.default_tts_voice
        model = model or self.default_tts_model
        logger.info(f"Recording a voice message with model {model}...")
        response = await self.client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            instructions=OPENAI_TTS_INSTRUCTIONS,
        )
        return await response.aread()

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

    def get_model_display_name(self, model_name: str) -> str:
        if "dall" in model_name:
            return model_name.replace("dall-e-", "DALL·E ")
        return model_name.replace("-", " ")
