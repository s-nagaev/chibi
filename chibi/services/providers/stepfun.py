import base64
import json
from io import BytesIO
from typing import Any

import httpx
from loguru import logger

from chibi.config import application_settings, gpt_settings
from chibi.exceptions import (
    NoApiKeyProvidedError,
    NotAuthorizedError,
    ServiceRateLimitError,
    ServiceResponseError,
)
from chibi.services.providers.provider import OpenAIFriendlyProvider, RestApiFriendlyProvider


class StepFun(OpenAIFriendlyProvider, RestApiFriendlyProvider):
    api_key = gpt_settings.stepfun_key
    chat_ready = True
    moderation_ready = True
    stt_ready = True
    tts_ready = True
    vision_ready = True

    name = "StepFun"
    base_url = "https://api.stepfun.ai/v1"
    model_name_keywords = ["step"]
    model_name_keywords_exclude = ["audio"]

    default_model = "step-5-preview"
    default_moderation_model = "step-3.5-flash-2603"
    default_stt_model = "stepaudio-2.5-asr"
    default_tts_model = "stepaudio-3-tts"
    default_tts_voice = "magnetic-voiced-male"
    default_vision_model = "step-3.7-flash"

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
            "Accept": "text/event-stream",
        }

    def _build_asr_payload(self, audio: BytesIO, model: str) -> dict[str, Any]:
        """Build the request body for the ASR SSE endpoint.

        Args:
            audio: The audio recording as a byte stream.
            model: The transcription model to use.

        Returns:
            The JSON-serializable request body. Base64 encoding of the raw
            audio bytes is expected here; the field names are confirmed
            against the live API during contract verification.
        """
        return {
            "model": model,
            "file": base64.b64encode(audio.getvalue()).decode("utf-8"),
        }

    def _process_asr_event(self, event_lines: list[str], chunks: list[str]) -> str | None:
        """Parse one buffered SSE event and update the delta accumulator.

        Args:
            event_lines: Raw SSE lines of a single event (everything between
                two blank separator lines).
            chunks: The running list of transcript text deltas; mutated when
                the event is a ``transcript.text.delta``.

        Returns:
            The final transcript text when the event is
            ``transcript.text.done``, or None otherwise.

        Raises:
            ValueError: When the event is an error event.
        """
        data_lines = [line[len("data:") :].strip() for line in event_lines if line.startswith("data:")]
        if not data_lines:
            return None
        try:
            payload = json.loads("\n".join(data_lines))
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None

        event_type = payload.get("type")
        if event_type == "transcript.text.delta":
            delta = payload.get("delta")
            if isinstance(delta, str):
                chunks.append(delta)
            return None
        if event_type == "transcript.text.done":
            text = payload.get("text")
            return text if isinstance(text, str) else ""
        if event_type == "error":
            raise ValueError("Could not transcribe audio message")
        return None

    async def _consume_asr_stream(self, response: httpx.Response) -> str:
        """Consume the ASR SSE stream incrementally and return the transcript.

        Args:
            response: The streaming HTTP response of the ASR call.

        Returns:
            The final transcript text from ``transcript.text.done``.

        Raises:
            ValueError: When the stream carries an error event, or ends
                without a ``transcript.text.done`` event.
        """
        chunks: list[str] = []
        final_text: str | None = None
        event_lines: list[str] = []

        async for line in response.aiter_lines():
            if line:
                event_lines.append(line)
                continue
            text = self._process_asr_event(event_lines, chunks)
            if text is not None:
                final_text = text
            event_lines = []
        if event_lines:
            text = self._process_asr_event(event_lines, chunks)
            if text is not None:
                final_text = text

        if final_text is None:
            raise ValueError("Could not transcribe audio message")
        return final_text

    async def transcribe(self, audio: BytesIO, model: str | None = None) -> str:
        """Transcribe an audio recording with StepFun's ASR SSE API.

        StepFun ASR does not use the OpenAI ``/v1/audio/transcriptions``
        endpoint, so the call goes through raw ``httpx``: a JSON
        ``POST {base_url}/audio/asr/sse`` carrying the Base64-encoded audio,
        read as a ``text/event-stream``. The stream is consumed
        incrementally: ``transcript.text.delta`` events are accumulated and
        ``transcript.text.done`` carries the final text; unknown or keepalive
        event types are skipped silently.

        Args:
            audio: The audio recording as a byte stream (OGG/Opus from
                Telegram is supported).
            model: The transcription model to use; falls back to the
                provider default when None.

        Returns:
            The transcribed text.

        Raises:
            NoApiKeyProvidedError: When no StepFun API key is configured.
            NotAuthorizedError: When the API rejects the key (HTTP 401).
            ServiceRateLimitError: When the API rate-limits the request (HTTP 429).
            ServiceResponseError: On any other HTTP or transport failure
                (connection errors, read timeouts, non-200 responses).
            ValueError: When the stream carries an error event, or the
                transcript is empty or missing its final text.
        """
        model = model or self.default_stt_model
        if not model:
            raise ValueError("Could not transcribe audio message")
        logger.info(f"[{self.name}] Transcribing audio with model {model}...")

        if not self.token:
            raise NoApiKeyProvidedError(provider=self.name)

        url = f"{self.base_url}/audio/asr/sse"
        try:
            async with self.get_async_httpx_client() as client:
                async with client.stream(
                    method="POST",
                    url=url,
                    json=self._build_asr_payload(audio=audio, model=model),
                    headers=self._headers,
                ) as response:
                    if response.status_code == 401:
                        raise NotAuthorizedError(provider=self.name, model=model)
                    if response.status_code == 429:
                        raise ServiceRateLimitError(provider=self.name, model=model)
                    if response.status_code != 200:
                        await response.aread()
                        logger.error(
                            f"Unexpected response from {self.name} API. "
                            f"Status code: {response.status_code}. Data: {response.text}"
                        )
                        raise ServiceResponseError(
                            provider=self.name,
                            model=model,
                            detail=f"status_code: {response.status_code}, body: {response.text[:200]}",
                        )
                    text = await self._consume_asr_stream(response)
        except (
            httpx.ConnectError,
            httpx.PoolTimeout,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
        ) as e:
            logger.error(f"An error occurred while calling the {self.name} API: {e}")
            raise ServiceResponseError(provider=self.name, model=model, detail=str(e)) from e

        if not text:
            raise ValueError("Could not transcribe audio message")

        if application_settings.log_prompt_data:
            logger.info(f"Transcribed text: {text}")
        return text
