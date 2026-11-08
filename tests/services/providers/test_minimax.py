from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest
from anthropic import BadRequestError

from chibi.exceptions import ContextLengthExceededError, ServiceResponseError
from chibi.services.providers import Minimax
from tests.conftest import MockResponse


@pytest.mark.asyncio
async def test_speech(minimax: Mock) -> None:
    provider = Minimax("test_token")

    data = {"data": {"audio": ""}}

    with patch(
        "chibi.services.providers.provider.RestApiFriendlyProvider._request", return_value=MockResponse(payload=data)
    ) as _request:
        result = await provider.speech(text="Hello world!", voice="test_voice", model="some_tts_model")

    assert result == b""
    _request.assert_called_once_with(
        method="POST",
        url="https://api.minimax.io/v1/t2a_v2",
        data={
            "model": "some_tts_model",
            "text": "Hello world!",
            "voice_setting": {"voice_id": "test_voice", "emotion": "happy", "speed": 1.2},
        },
    )


@pytest.mark.asyncio
async def test_request_fallback_error_includes_status_and_body() -> None:
    provider = Minimax("test_token")
    body = (
        '{"type":"error","error":{"message":"invalid params, this model is not supported'
        ' by /v2/video_generation (2013)"}}'
    )
    http_response = httpx.Response(status_code=400, text=body)
    client_mock = MagicMock()
    client_mock.__aenter__.return_value = client_mock
    client_mock.__aexit__.return_value = None
    client_mock.request = AsyncMock(return_value=http_response)

    with patch.object(Minimax, "get_async_httpx_client", return_value=client_mock):
        with pytest.raises(ServiceResponseError) as exc_info:
            await provider._request(method="POST", url="https://api.minimax.io/v2/video_generation", data={})

    detail = exc_info.value.detail
    assert "status_code: 400" in detail
    assert "invalid params" in detail


@pytest.mark.asyncio
async def test_request_fallback_error_truncates_body_to_200_chars() -> None:
    provider = Minimax("test_token")
    body = "x" * 500
    http_response = httpx.Response(status_code=500, text=body)
    client_mock = MagicMock()
    client_mock.__aenter__.return_value = client_mock
    client_mock.__aexit__.return_value = None
    client_mock.request = AsyncMock(return_value=http_response)

    with patch.object(Minimax, "get_async_httpx_client", return_value=client_mock):
        with pytest.raises(ServiceResponseError) as exc_info:
            await provider._request(method="POST", url="https://api.minimax.io/v2/video_generation", data={})

    assert len(exc_info.value.detail) < 250


def _make_anthropic_bad_request_error(message: str) -> BadRequestError:
    """Build an Anthropic-compatible BadRequestError carrying the given body message."""
    request = httpx.Request("POST", "https://api.minimax.io/anthropic/v1/messages")
    response = httpx.Response(
        400,
        request=request,
        json={"type": "error", "error": {"type": "invalid_request_error", "message": message}},
    )
    return BadRequestError("Error code: 400", response=response, body=response.json())


@pytest.mark.asyncio
async def test_minimax_context_overflow_bad_request_raises_context_length_exceeded_error() -> None:
    """The MiniMax (AnthropicFriendly) path maps an overflow 400 to ContextLengthExceededError."""
    provider = Minimax(token="test-token")
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(
        side_effect=_make_anthropic_bad_request_error("prompt is too long: 250000 tokens > 200000 maximum")
    )
    provider._client = mock_client

    with pytest.raises(ContextLengthExceededError):
        await provider._generate_content(
            model="MiniMax-M3",
            system_prompt="system",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
        )

    assert mock_client.messages.create.await_count == 1
