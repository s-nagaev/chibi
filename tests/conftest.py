from typing import AsyncIterator, Generator, Iterable
from unittest.mock import AsyncMock, Mock, patch

import pytest


class AsyncBytesIterator:
    def __init__(self, bytes_to_iter: Iterable[bytes]) -> None:
        self.bytes_to_iter = self._bytes_to_iter(bytes_to_iter)

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self

    async def __anext__(self) -> bytes:
        return await self.bytes_to_iter.__anext__()

    @staticmethod
    async def _bytes_to_iter(bytes_to_iter: Iterable[bytes]) -> AsyncIterator[bytes]:
        for byte in bytes_to_iter:
            yield byte


class AsyncElevenLabsMock:
    def __init__(self, token: str) -> None:
        self.token = token
        self._text_to_speech: Mock | None = None
        self._speech_to_text: Mock | None = None
        self._generate_music: Mock | None = None

    @property
    def text_to_speech(self) -> Mock:
        if self._text_to_speech:
            return self._text_to_speech
        self._text_to_speech = Mock()
        self._text_to_speech.convert = Mock(return_value=AsyncBytesIterator([b"Hello", b"world!"]))
        return self._text_to_speech

    @property
    def speech_to_text(self) -> Mock:
        if self._speech_to_text:
            return self._speech_to_text
        self._speech_to_text = Mock()
        self._speech_to_text.convert = AsyncMock(return_value=Mock(text="Hello world!"))
        return self._speech_to_text

    @property
    def music(self) -> Mock:
        if self._generate_music:
            return self._generate_music
        self._generate_music = Mock()
        self._generate_music.compose = Mock(return_value=AsyncBytesIterator([b"Hello", b"world!"]))
        return self._generate_music


@pytest.fixture
def eleven_labs() -> Generator:
    _eleven_labs = AsyncElevenLabsMock("test_token")
    with patch("chibi.services.providers.eleven_labs.AsyncElevenLabs", return_value=_eleven_labs) as mock_tts_convert:
        yield mock_tts_convert
