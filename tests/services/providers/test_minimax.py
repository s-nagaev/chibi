from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest

from chibi.exceptions import ServiceResponseError
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
