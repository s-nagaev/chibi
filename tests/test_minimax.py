from unittest.mock import Mock, patch

import pytest

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
        data={"model": "some_tts_model", "text": "Hello world!", "voice_setting": {"voice_id": "test_voice"}},
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": "Bearer test_token",
        },
    )
