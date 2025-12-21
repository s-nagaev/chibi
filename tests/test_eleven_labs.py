from io import BytesIO
from unittest.mock import Mock, call

import pytest

from chibi.services.providers import ElevenLabs


@pytest.mark.asyncio
async def test_speach(eleven_labs: Mock) -> None:
    provider = ElevenLabs("test_token")
    result = await provider.speech(text="Hello world!", voice="test_voice", model="some_tts_model")

    assert result == b"Helloworld!"
    assert eleven_labs.call_args_list == [call(api_key="test_token")]
    provider.client.text_to_speech.convert.assert_called_once_with(
        text="Hello world!",
        voice_id="test_voice",
        model_id="some_tts_model",
        output_format=ElevenLabs.output_format,
    )


@pytest.mark.asyncio
async def test_transcribe(eleven_labs: Mock) -> None:
    provider = ElevenLabs("test_token")
    audio = BytesIO()
    result = await provider.transcribe(audio=audio, model="some_stt_model")

    assert result == "Hello world!"
    assert eleven_labs.call_args_list == [call(api_key="test_token")]
    provider.client.speech_to_text.convert.assert_called_once_with(
        file=audio,
        model_id="some_stt_model",
        tag_audio_events=ElevenLabs.tag_audio_events,
        language_code=ElevenLabs.language_code,
    )


@pytest.mark.asyncio
async def test_generate_music(eleven_labs: Mock) -> None:
    provider = ElevenLabs("test_token")
    result = await provider.generate_music(prompt="Some prompt", music_length_ms=1000)

    assert result == b"Helloworld!"
    assert eleven_labs.call_args_list == [call(api_key="test_token")]
    provider.client.music.compose.assert_called_once_with(prompt="Some prompt", music_length_ms=1000)
