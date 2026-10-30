from io import BytesIO

import httpx
import pytest

from chibi.exceptions import (
    NoApiKeyProvidedError,
    NotAuthorizedError,
    ServiceRateLimitError,
    ServiceResponseError,
)
from chibi.services.providers.eleven_labs import ElevenLabs
from chibi.services.providers.mistralai_native import _TRANSCRIPTIONS_URL, MistralAI
from chibi.services.providers.provider import RegisteredProviders

TEST_TOKEN = "test-token-for-stt"
SAMPLE_AUDIO = b"fake-ogg-audio-bytes"
DEFAULT_STT_MODEL = "voxtral-mini-transcribe-3-0"
CUSTOM_STT_MODEL = "voxtral-small-transcribe-3-0"


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
    return httpx.Response(200, json={"model": DEFAULT_STT_MODEL, "text": "Hello there"})


async def test_transcribe_happy_path():
    provider = _make_provider()
    requests = _install_transport(provider, _ok)

    result = await provider.transcribe(audio=BytesIO(SAMPLE_AUDIO))

    assert result == "Hello there"
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == _TRANSCRIPTIONS_URL
    assert request.method == "POST"
    assert request.headers["Authorization"] == f"Bearer {TEST_TOKEN}"


async def test_transcribe_sends_model_and_file_as_multipart_fields():
    provider = _make_provider()
    requests = _install_transport(provider, _ok)

    await provider.transcribe(audio=BytesIO(SAMPLE_AUDIO))

    body = requests[0].content
    assert b'name="model"' in body
    assert DEFAULT_STT_MODEL.encode() in body
    assert b'name="file"' in body
    assert b'filename="voice.ogg"' in body
    assert SAMPLE_AUDIO in body


async def test_transcribe_custom_model_override():
    provider = _make_provider()
    requests = _install_transport(provider, _ok)

    result = await provider.transcribe(
        audio=BytesIO(SAMPLE_AUDIO),
        model=CUSTOM_STT_MODEL,
    )

    assert result == "Hello there"
    assert CUSTOM_STT_MODEL.encode() in requests[0].content
    assert DEFAULT_STT_MODEL.encode() not in requests[0].content


def test_transcribe_default_model_attribute():
    provider = _make_provider()

    assert provider.stt_ready is True
    assert provider.default_stt_model == DEFAULT_STT_MODEL


async def test_transcribe_unauthorized():
    provider = _make_provider()

    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "Unauthorized"})

    _install_transport(provider, responder)

    with pytest.raises(NotAuthorizedError):
        await provider.transcribe(audio=BytesIO(SAMPLE_AUDIO))


async def test_transcribe_rate_limited():
    provider = _make_provider()

    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"message": "Too many requests"})

    _install_transport(provider, responder)

    with pytest.raises(ServiceRateLimitError):
        await provider.transcribe(audio=BytesIO(SAMPLE_AUDIO))


async def test_transcribe_other_http_error():
    provider = _make_provider()

    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"message": "Internal error"})

    _install_transport(provider, responder)

    with pytest.raises(ServiceResponseError):
        await provider.transcribe(audio=BytesIO(SAMPLE_AUDIO))


async def test_transcribe_transport_failure():
    provider = _make_provider()

    def responder(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    _install_transport(provider, responder)

    with pytest.raises(ServiceResponseError):
        await provider.transcribe(audio=BytesIO(SAMPLE_AUDIO))


async def test_transcribe_empty_text_raises_value_error():
    provider = _make_provider()

    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": DEFAULT_STT_MODEL, "text": ""})

    _install_transport(provider, responder)

    with pytest.raises(ValueError, match="Could not transcribe audio message"):
        await provider.transcribe(audio=BytesIO(SAMPLE_AUDIO))


async def test_transcribe_without_api_key():
    provider = _make_provider(token="")

    with pytest.raises(NoApiKeyProvidedError):
        await provider.transcribe(audio=BytesIO(SAMPLE_AUDIO))


def test_default_stt_fallback_provider_is_elevenlabs():
    """Without gpt_settings.stt_provider the fallback is the first registered stt_ready provider.

    ``first_stt_ready`` iterates ``RegisteredProviders.available`` (insertion
    order == registration order == import order), so the earliest imported
    stt_ready provider wins. ElevenLabs is imported before MistralAI in
    ``chibi/services/providers/__init__.py``.
    """
    order = list(RegisteredProviders.all)
    assert order.index("elevenlabs") < order.index("mistralai")
    assert ElevenLabs.stt_ready is True
    assert MistralAI.stt_ready is True
