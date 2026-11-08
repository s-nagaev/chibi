"""Unit tests for the StepFun provider adapter."""

import asyncio
import base64
import json
from io import BytesIO
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from chibi.config import gpt_settings
from chibi.exceptions import (
    NoApiKeyProvidedError,
    NotAuthorizedError,
    ServiceRateLimitError,
    ServiceResponseError,
)
from chibi.schemas.app import ModelChangeSchema
from chibi.services.providers import RegisteredProviders, StepFun

TEST_TOKEN = "test-stepfun-token"
SAMPLE_AUDIO = b"fake-ogg-audio-bytes"
ASR_URL = "https://api.stepfun.ai/v1/audio/asr/sse"
DEFAULT_STT_MODEL = "stepaudio-2.5-asr"
CUSTOM_STT_MODEL = "stepaudio-2.5-asr-custom"
CHAT_MODEL_IDS = ["step-5-preview", "step-3.7-flash", "step-3.5-flash", "step-3.5-flash-2603"]
AUDIO_MODEL_IDS = ["stepaudio-2.5-asr", "stepaudio-3-tts"]


def _make_provider(token: str = TEST_TOKEN) -> StepFun:
    """Return a provider instance wired with the test token."""
    return StepFun(token=token)


def _sse(*payloads: dict[str, Any]) -> bytes:
    """Encode payloads as an SSE stream (one ``data:`` line per event)."""
    return "".join(f"data: {json.dumps(payload)}\n\n" for payload in payloads).encode()


def _install_transport(provider: StepFun, responder) -> list[httpx.Request]:
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


def _ok_stream(body: bytes) -> httpx.Response:
    """Return a 200 SSE response carrying the given raw body."""
    return httpx.Response(
        200,
        headers={"Content-Type": "text/event-stream"},
        content=body,
    )


async def _respond_400(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Consume one HTTP request and answer with a 400 carrying a JSON body."""
    body = b'{"message": "invalid audio payload"}'
    while True:
        line = await reader.readline()
        if line in (b"\r\n", b"\n", b""):
            break
    head = (
        "HTTP/1.1 400 Bad Request\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n\r\n"
    ).encode()
    writer.write(head + body)
    await writer.drain()
    writer.close()


def test_stepfun_registered_in_registered_providers() -> None:
    """The class is registered under its lowercase provider name on import."""
    assert RegisteredProviders.all["stepfun"] is StepFun


def test_availability_gated_on_api_key() -> None:
    """The provider is only marked available when a truthy api_key is configured."""
    assert StepFun.api_key == gpt_settings.stepfun_key
    if StepFun.api_key:
        assert "stepfun" in RegisteredProviders.available
    else:
        assert "stepfun" not in RegisteredProviders.available


def test_provider_attributes() -> None:
    """Full chat/vision/tts/stt/moderation readiness and default model values."""
    assert StepFun.name == "StepFun"
    assert StepFun.base_url == "https://api.stepfun.ai/v1"
    assert StepFun.chat_ready is True
    assert StepFun.vision_ready is True
    assert StepFun.tts_ready is True
    assert StepFun.stt_ready is True
    assert StepFun.moderation_ready is True
    assert StepFun.ocr_ready is False
    assert StepFun.image_generation_ready is False
    assert StepFun.default_model == "step-5-preview"
    assert StepFun.default_vision_model == "step-3.7-flash"
    assert StepFun.default_tts_model == "stepaudio-3-tts"
    assert StepFun.default_tts_voice == "magnetic-voiced-male"
    assert StepFun.default_moderation_model == "step-3.5-flash-2603"
    assert StepFun.default_stt_model == DEFAULT_STT_MODEL
    assert StepFun.model_name_keywords == ["step"]
    assert StepFun.model_name_keywords_exclude == ["audio"]


def test_model_filtering_chat_ready_models() -> None:
    """Chat filtering keeps every ``step-`` chat id and drops all ``stepaudio-*`` ids."""
    provider = _make_provider()
    models = [
        ModelChangeSchema(provider="StepFun", name=name, image_generation=False)
        for name in CHAT_MODEL_IDS + AUDIO_MODEL_IDS
    ]

    with patch("chibi.services.providers.provider.gpt_settings") as mock_settings:
        mock_settings.models_whitelist = []
        mock_settings.models_blacklist = []
        filtered = provider.filter_and_return_list_of_models(models)

    assert [model.name for model in filtered] == [
        "step-5-preview",
        "step-3.7-flash",
        "step-3.5-flash-2603",
        "step-3.5-flash",
    ]
    assert not any("stepaudio" in model.name for model in filtered)


def test_model_filtering_image_generation() -> None:
    """Only models flagged for image generation survive the image_generation filter."""
    provider = _make_provider()
    models = [
        ModelChangeSchema(provider="StepFun", name="step-image-model", image_generation=True),
        ModelChangeSchema(provider="StepFun", name="step-5-preview", image_generation=False),
    ]

    with patch("chibi.services.providers.provider.gpt_settings") as mock_settings:
        mock_settings.models_whitelist = []
        mock_settings.models_blacklist = []
        filtered = provider.filter_and_return_list_of_models(models, image_generation=True)

    assert [model.name for model in filtered] == ["step-image-model"]


@pytest.mark.asyncio
async def test_transcribe_happy_path() -> None:
    """Delta events accumulate and the final ``transcript.text.done`` carries the text."""
    provider = _make_provider()
    stream = _sse(
        {"type": "transcript.text.delta", "delta": "Hello "},
        {"type": "transcript.text.delta", "delta": "there"},
        {"type": "transcript.text.done", "text": "Hello there"},
    )
    requests = _install_transport(provider, lambda request: _ok_stream(stream))

    result = await provider.transcribe(audio=BytesIO(SAMPLE_AUDIO))

    assert result == "Hello there"
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == ASR_URL
    assert request.method == "POST"
    assert request.headers["Authorization"] == f"Bearer {TEST_TOKEN}"
    body = json.loads(request.content)
    assert body["model"] == DEFAULT_STT_MODEL
    assert body["file"] == base64.b64encode(SAMPLE_AUDIO).decode("utf-8")


@pytest.mark.asyncio
async def test_transcribe_custom_model_override() -> None:
    """An explicit model argument replaces the provider default in the payload."""
    provider = _make_provider()
    stream = _sse(
        {"type": "transcript.text.done", "text": "Hello there"},
    )
    requests = _install_transport(provider, lambda request: _ok_stream(stream))

    result = await provider.transcribe(audio=BytesIO(SAMPLE_AUDIO), model=CUSTOM_STT_MODEL)

    assert result == "Hello there"
    assert json.loads(requests[0].content)["model"] == CUSTOM_STT_MODEL


@pytest.mark.asyncio
async def test_transcribe_keepalive_and_unknown_events_skipped() -> None:
    """Comment keepalives and unknown event types are skipped without aborting the stream."""
    provider = _make_provider()

    def responder(request: httpx.Request) -> httpx.Response:
        raw = (
            b": keepalive\n\n"
            + _sse({"type": "heartbeat", "at": 1})
            + _sse({"type": "transcript.text.delta", "delta": "Hi"})
            + _sse({"type": "transcript.text.done", "text": "Hi"})
        )
        return _ok_stream(raw)

    _install_transport(provider, responder)

    assert await provider.transcribe(audio=BytesIO(SAMPLE_AUDIO)) == "Hi"


@pytest.mark.asyncio
async def test_transcribe_in_stream_error_event_raises_value_error() -> None:
    """An ``error`` event in the stream surfaces as ValueError."""
    provider = _make_provider()
    stream = _sse(
        {"type": "transcript.text.delta", "delta": "Hel"},
        {"type": "error", "code": "asr_failed"},
    )
    _install_transport(provider, lambda request: _ok_stream(stream))

    with pytest.raises(ValueError, match="Could not transcribe audio message"):
        await provider.transcribe(audio=BytesIO(SAMPLE_AUDIO))


@pytest.mark.asyncio
async def test_transcribe_empty_final_text_raises_value_error() -> None:
    """A ``transcript.text.done`` event with empty text raises ValueError."""
    provider = _make_provider()
    stream = _sse(
        {"type": "transcript.text.delta", "delta": "ignored"},
        {"type": "transcript.text.done", "text": ""},
    )
    _install_transport(provider, lambda request: _ok_stream(stream))

    with pytest.raises(ValueError, match="Could not transcribe audio message"):
        await provider.transcribe(audio=BytesIO(SAMPLE_AUDIO))


@pytest.mark.asyncio
async def test_transcribe_missing_done_event_raises_value_error() -> None:
    """A stream that ends with deltas only (no ``transcript.text.done``) raises ValueError."""
    provider = _make_provider()
    stream = _sse({"type": "transcript.text.delta", "delta": "partial"})
    _install_transport(provider, lambda request: _ok_stream(stream))

    with pytest.raises(ValueError, match="Could not transcribe audio message"):
        await provider.transcribe(audio=BytesIO(SAMPLE_AUDIO))


@pytest.mark.asyncio
async def test_transcribe_unauthorized() -> None:
    """HTTP 401 maps to NotAuthorizedError."""
    provider = _make_provider()

    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "Unauthorized"})

    _install_transport(provider, responder)

    with pytest.raises(NotAuthorizedError):
        await provider.transcribe(audio=BytesIO(SAMPLE_AUDIO))


@pytest.mark.asyncio
async def test_transcribe_rate_limited() -> None:
    """HTTP 429 maps to ServiceRateLimitError."""
    provider = _make_provider()

    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"message": "Too many requests"})

    _install_transport(provider, responder)

    with pytest.raises(ServiceRateLimitError):
        await provider.transcribe(audio=BytesIO(SAMPLE_AUDIO))


@pytest.mark.asyncio
async def test_transcribe_non_200_via_mock_transport() -> None:
    """A 400 with a body maps to ServiceResponseError with the body inside detail."""
    provider = _make_provider()

    def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"message": "invalid audio payload"})

    _install_transport(provider, responder)

    with pytest.raises(ServiceResponseError) as exc_info:
        await provider.transcribe(audio=BytesIO(SAMPLE_AUDIO))

    assert "status_code: 400" in exc_info.value.detail
    assert "invalid audio payload" in exc_info.value.detail


@pytest.mark.asyncio
async def test_transcribe_non_200_real_socket_returns_body_in_detail() -> None:
    """A real local HTTP server answering 400 yields ServiceResponseError with the body.

    MockTransport serves pre-built responses and structurally cannot expose a
    streaming-response body that was never read, so the non-200 branch is
    additionally exercised over a real localhost socket.
    """
    server = await asyncio.start_server(_respond_400, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    provider = _make_provider()
    provider.base_url = f"http://127.0.0.1:{port}/v1"
    try:
        with pytest.raises(ServiceResponseError) as exc_info:
            await provider.transcribe(audio=BytesIO(SAMPLE_AUDIO))
    finally:
        server.close()
        await server.wait_closed()

    assert "status_code: 400" in exc_info.value.detail
    assert "invalid audio payload" in exc_info.value.detail


@pytest.mark.parametrize(
    "transport_error",
    [
        httpx.ConnectError("connection refused"),
        httpx.PoolTimeout("pool exhausted"),
        httpx.ReadTimeout("timed out"),
        httpx.RemoteProtocolError("server disconnected"),
    ],
    ids=["connect-error", "pool-timeout", "read-timeout", "remote-protocol-error"],
)
@pytest.mark.asyncio
async def test_transcribe_transport_errors_mapped_to_service_response_error(
    transport_error: httpx.HTTPError,
) -> None:
    """Transport failures map to ServiceResponseError with the original message in detail."""
    provider = _make_provider()

    def responder(request: httpx.Request) -> httpx.Response:
        raise transport_error

    _install_transport(provider, responder)

    with pytest.raises(ServiceResponseError) as exc_info:
        await provider.transcribe(audio=BytesIO(SAMPLE_AUDIO))

    assert str(transport_error) in exc_info.value.detail


@pytest.mark.asyncio
async def test_transcribe_without_api_key() -> None:
    """An empty token short-circuits with NoApiKeyProvidedError before any HTTP call."""
    provider = _make_provider(token="")

    with pytest.raises(NoApiKeyProvidedError):
        await provider.transcribe(audio=BytesIO(SAMPLE_AUDIO))
