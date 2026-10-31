import base64
import json

import httpx
import pytest

from chibi.exceptions import (
    NoApiKeyProvidedError,
    NotAuthorizedError,
    ServiceRateLimitError,
    ServiceResponseError,
)
from chibi.services.providers.eleven_labs import ElevenLabs
from chibi.services.providers.mistralai_native import _TTS_URL, MistralAI
from chibi.services.providers.provider import RegisteredProviders

TEST_TOKEN = "test-token-for-tts"
SAMPLE_TEXT = "Your order shipped this morning."
DEFAULT_TTS_MODEL = "voxtral-mini-tts-3-0"
DEFAULT_TTS_VOICE = "gb_jane_neutral"
CUSTOM_TTS_MODEL = "voxtral-mini-tts-3"
CUSTOM_TTS_VOICE = "en_paul_neutral"
SAMPLE_AUDIO = b"fake-mp3-audio-bytes"


def _make_provider(token: str = TEST_TOKEN) -> MistralAI:
    return MistralAI(token=token)


def _install_transport(provider: MistralAI, responder) -> list[httpx.Request]:
    """Route the provider's httpx calls through a mock transport.

    Args:
        provider: The provider under test.
        responder: Callable turning an httpx.Request into an httpx.Response.

    Returns:
        The list of requests captured by the transport.
    """
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return responder(request)

    transport = httpx.MockTransport(handler)

    def get_client() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=transport)

    setattr(provider, "get_async_httpx_client", get_client)
    return requests


def _ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={"model": DEFAULT_TTS_MODEL, "audio_data": base64.b64encode(SAMPLE_AUDIO).decode()},
    )


def _request_json(request: httpx.Request) -> dict:
    return json.loads(request.content)


async def test_speech_happy_path_decodes_base64():
    provider = _make_provider()
    requests = _install_transport(provider, _ok)

    result = await provider.speech(text=SAMPLE_TEXT)

    assert result == SAMPLE_AUDIO
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == _TTS_URL
    assert request.method == "POST"
    assert request.headers["Authorization"] == f"Bearer {TEST_TOKEN}"


async def test_speech_sends_json_body_with_default_model_voice_and_mp3():
    provider = _make_provider()
    requests = _install_transport(provider, _ok)

    await provider.speech(text=SAMPLE_TEXT)

    body = _request_json(requests[0])
    assert body == {
        "model": DEFAULT_TTS_MODEL,
        "input": SAMPLE_TEXT,
        "voice_id": DEFAULT_TTS_VOICE,
        "response_format": "mp3",
    }


async def test_speech_custom_model_and_voice_override():
    provider = _make_provider()
    requests = _install_transport(provider, _ok)

    result = await provider.speech(
        text=SAMPLE_TEXT,
        voice=CUSTOM_TTS_VOICE,
        model=CUSTOM_TTS_MODEL,
    )

    assert result == SAMPLE_AUDIO
    body = _request_json(requests[0])
    assert body["model"] == CUSTOM_TTS_MODEL
    assert body["voice_id"] == CUSTOM_TTS_VOICE


def test_speech_default_model_and_voice_attributes():
    provider = _make_provider()

    assert provider.tts_ready is True
    assert provider.default_tts_model == DEFAULT_TTS_MODEL
    assert provider.default_tts_voice == DEFAULT_TTS_VOICE


async def test_speech_unauthorized():
    provider = _make_provider()

    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "Unauthorized"})

    _install_transport(provider, responder)

    with pytest.raises(NotAuthorizedError):
        await provider.speech(text=SAMPLE_TEXT)


async def test_speech_rate_limited():
    provider = _make_provider()

    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"message": "Too many requests"})

    _install_transport(provider, responder)

    with pytest.raises(ServiceRateLimitError):
        await provider.speech(text=SAMPLE_TEXT)


async def test_speech_other_http_error():
    provider = _make_provider()

    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"message": "Internal error"})

    _install_transport(provider, responder)

    with pytest.raises(ServiceResponseError):
        await provider.speech(text=SAMPLE_TEXT)


async def test_speech_transport_failure():
    provider = _make_provider()

    def responder(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    _install_transport(provider, responder)

    with pytest.raises(ServiceResponseError):
        await provider.speech(text=SAMPLE_TEXT)


async def test_speech_missing_audio_data_raises_value_error():
    provider = _make_provider()

    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": DEFAULT_TTS_MODEL})

    _install_transport(provider, responder)

    with pytest.raises(ValueError, match="Could not generate speech audio"):
        await provider.speech(text=SAMPLE_TEXT)


async def test_speech_empty_audio_data_raises_value_error():
    provider = _make_provider()

    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": DEFAULT_TTS_MODEL, "audio_data": ""})

    _install_transport(provider, responder)

    with pytest.raises(ValueError, match="Could not generate speech audio"):
        await provider.speech(text=SAMPLE_TEXT)


async def test_speech_malformed_base64_raises_value_error():
    provider = _make_provider()

    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": DEFAULT_TTS_MODEL, "audio_data": "!!!not-base64!!!"})

    _install_transport(provider, responder)

    with pytest.raises(ValueError, match="Could not generate speech audio"):
        await provider.speech(text=SAMPLE_TEXT)


async def test_speech_without_api_key():
    provider = _make_provider(token="")

    with pytest.raises(NoApiKeyProvidedError):
        await provider.speech(text=SAMPLE_TEXT)


def test_default_tts_fallback_provider_is_elevenlabs():
    """Without gpt_settings.tts_provider the fallback is the first registered tts_ready provider.

    ``first_tts_ready`` iterates ``RegisteredProviders.available`` (insertion
    order == registration order == import order), so the earliest imported
    tts_ready provider wins. ElevenLabs is imported before MistralAI in
    ``chibi/services/providers/__init__.py``.
    """
    order = list(RegisteredProviders.all)
    assert order.index("elevenlabs") < order.index("mistralai")
    assert ElevenLabs.tts_ready is True
    assert MistralAI.tts_ready is True
