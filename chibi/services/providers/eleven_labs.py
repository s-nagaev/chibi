from io import BytesIO

from elevenlabs.client import AsyncElevenLabs
from loguru import logger

from chibi.config import application_settings, gpt_settings
from chibi.exceptions import NoApiKeyProvidedError
from chibi.services.providers.provider import Provider


class ElevenLabs(Provider):
    api_key = gpt_settings.elevenlabs_api_key
    chat_ready = False
    tts_ready = True
    stt_ready = True

    name = "ElevenLabs"
    default_stt_model = "scribe_v1"
    tag_audio_events = False
    language_code = None
    default_tts_voice = "JBFqnCBsd6RMkjVDRZzb"
    default_tts_model = "eleven_multilingual_v2"
    output_format = "mp3_44100_128"
    music_length_ms = 10000

    def __init__(self, token: str) -> None:
        self._client: AsyncElevenLabs | None = None
        super().__init__(token=token)

    @property
    def client(self) -> AsyncElevenLabs:
        if self._client:
            return self._client

        if not self.token:
            raise NoApiKeyProvidedError(provider=self.name)

        self._client = AsyncElevenLabs(
            api_key=self.token,
        )

        return self._client

    async def speech(self, text: str, voice: str = default_tts_voice, model: str = default_tts_model) -> bytes:
        logger.info(f"Recording a voice message with model {model}...")
        response = await self.client.text_to_speech.convert(
            text=text,
            voice_id=voice,
            model_id=model,
            output_format=self.output_format,
        )

        buf = bytearray()
        async for chunk in response:
            buf.extend(chunk)

        return bytes(buf)

    async def transcribe(self, audio: BytesIO, model: str = default_stt_model) -> str:
        logger.info(f"Transcribing audio with model {model}...")
        response = await self.client.speech_to_text.convert(
            file=audio, model_id=model, tag_audio_events=self.tag_audio_events, language_code=self.language_code
        )
        if response:
            if application_settings.log_prompt_data:
                logger.info(f"Transcribed text: {response.text}")
            return response.text
        raise ValueError("Could not transcribe audio message")

    async def generate_music(self, prompt: str, music_length_ms: int = music_length_ms) -> bytes:
        logger.info(f"Generating music for {music_length_ms} ms...")
        response = self.client.music.compose(prompt=prompt, music_length_ms=music_length_ms)

        buf = bytearray()
        async for chunk in response:
            buf.extend(chunk)

        return bytes(buf)
