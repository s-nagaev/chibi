"""Tests for the save_path wiring and honest responses of the media tools."""

import inspect
from collections.abc import Generator
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest

from chibi.config import gpt_settings
from chibi.schemas.app import VideoResult
from chibi.services.interface import UserInterface
from chibi.services.providers.tools import utils as tools_utils
from chibi.services.providers.tools.exceptions import ToolException
from chibi.services.providers.tools.media import (
    GenerateAdvancedMusicViaSunoTool,
    GenerateImageTool,
    GenerateMusicViaElevenLabsTool,
    GenerateMusicViaMinimaxTool,
    GenerateMusicViaSunoTool,
    TextToSpeechTool,
)
from chibi.services.providers.tools.video import GenerateVideoTool


class StubInterface(UserInterface):
    supports_binary_media = False

    send_audio: AsyncMock
    send_video: AsyncMock
    send_images: AsyncMock
    send_document: AsyncMock

    def __init__(self) -> None:
        self.send_audio = AsyncMock()
        self.send_video = AsyncMock()
        self.send_images = AsyncMock()
        self.send_document = AsyncMock()

    @property
    def chat_id(self) -> int:
        return 42

    @property
    def user_id(self) -> int:
        return 7


class DeliveringInterface(StubInterface):
    supports_binary_media = True


@pytest.fixture
def media_dir(tmp_path: Path) -> Path:
    return tmp_path / "media-out"


@pytest.fixture
def patched_media_dir(monkeypatch: pytest.MonkeyPatch, media_dir: Path) -> Path:
    monkeypatch.setattr(tools_utils, "application_settings", SimpleNamespace(resolved_media_output_dir=str(media_dir)))
    return media_dir


@pytest.fixture
def filesystem_access_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpt_settings, "filesystem_access", True)


@pytest.fixture
def filesystem_access_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpt_settings, "filesystem_access", False)


@pytest.fixture
def tts_user() -> Generator[SimpleNamespace, None, None]:
    provider = SimpleNamespace(speech=AsyncMock(return_value=b"speech-bytes"))
    user = SimpleNamespace(tts_provider=provider)
    with patch("chibi.services.providers.tools.media.get_chibi_user", AsyncMock(return_value=user)):
        yield user


def _suno_version(audio_url: str, image_url: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        id=f"id-{audio_url}",
        title="Track",
        audio_url=audio_url,
        source_audio_url=None,
        stream_audio_url=None,
        source_stream_audio_url=None,
        source_image_url=None,
        image_url=image_url,
        duration=10,
    )


def _suno_poll_result() -> SimpleNamespace:
    return SimpleNamespace(
        model_dump=lambda: {"data": None},
        data=SimpleNamespace(
            response=SimpleNamespace(
                suno_data=[
                    _suno_version("https://example.com/a1.mp3", None),
                    _suno_version("https://example.com/a2.mp3", None),
                ]
            )
        ),
    )


@pytest.fixture
def suno_provider() -> Generator[SimpleNamespace, None, None]:
    provider = SimpleNamespace(
        order_music_generation=AsyncMock(return_value=123),
        order_music_generation_advanced_mode=AsyncMock(return_value=123),
        poll_result=AsyncMock(return_value=_suno_poll_result()),
    )
    with patch.object(GenerateMusicViaSunoTool, "_provider", provider):
        yield provider


@pytest.fixture
def elevenlabs_provider() -> Generator[SimpleNamespace, None, None]:
    provider = SimpleNamespace(generate_music=AsyncMock(return_value=b"track-bytes"))
    with patch.object(GenerateMusicViaElevenLabsTool, "_provider", provider):
        yield provider


@pytest.fixture
def image_patches(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    monkeypatch.setattr(
        "chibi.services.providers.tools.media.user_has_reached_images_generation_limit", AsyncMock(return_value=False)
    )
    generate_image_mock = AsyncMock(return_value=[BytesIO(b"png-bytes")])
    monkeypatch.setattr("chibi.services.providers.tools.media.generate_image", generate_image_mock)
    return generate_image_mock


@pytest.fixture
def video_patches(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    monkeypatch.setattr(
        "chibi.services.providers.tools.video.user_has_reached_videos_generation_limit", AsyncMock(return_value=False)
    )
    generate_video_mock = AsyncMock(return_value=VideoResult(video=b"mp4-bytes", thumbnail=None, duration=5))
    monkeypatch.setattr("chibi.services.providers.tools.video.generate_video", generate_video_mock)
    return generate_video_mock


class TestSchemaPlumbing:
    """save_path reaches both the static schema and the function signature."""

    @pytest.mark.parametrize(
        "tool_cls",
        [
            TextToSpeechTool,
            GenerateImageTool,
            GenerateVideoTool,
            GenerateMusicViaSunoTool,
            GenerateAdvancedMusicViaSunoTool,
            GenerateMusicViaElevenLabsTool,
        ],
    )
    def test_save_path_in_schema_and_signature(self, tool_cls: Any) -> None:
        parameters = cast(dict[str, Any], tool_cls.definition["function"]["parameters"])
        properties = cast(dict[str, Any], parameters["properties"])
        assert properties["save_path"]["type"] == "string"
        assert "save_path" not in parameters["required"]

        signature = inspect.signature(tool_cls.function)
        assert "save_path" in signature.parameters
        assert signature.parameters["save_path"].default is None

    def test_minimax_and_listing_tools_untouched(self) -> None:
        parameters = cast(dict[str, Any], GenerateMusicViaMinimaxTool.definition["function"]["parameters"])
        assert "save_path" not in parameters["properties"]


class TestTextToSpeechTool:
    """The TTS tool implements the response matrix."""

    async def test_delivering_runner_without_save_path_writes_nothing(
        self, tts_user: SimpleNamespace, patched_media_dir: Path, filesystem_access_enabled: Path
    ) -> None:
        interface = DeliveringInterface()
        response = await TextToSpeechTool.function(text="hello there", interface=interface)

        assert response == {"detail": "Audio was successfully sent."}
        interface.send_audio.assert_awaited_once()
        assert not patched_media_dir.exists()

    async def test_delivering_runner_with_save_path_saves_and_reports(
        self, tts_user: SimpleNamespace, tmp_path: Path, filesystem_access_enabled: Path
    ) -> None:
        interface = DeliveringInterface()
        target = tmp_path / "speech.mp3"
        response = await TextToSpeechTool.function(text="hello there", save_path=str(target), interface=interface)

        assert response["detail"] == "Audio was successfully sent."
        assert response["saved_to"] == [str(target.resolve())]
        assert target.read_bytes() == b"speech-bytes"
        interface.send_audio.assert_awaited_once()

    async def test_delivering_runner_save_failure_is_partial_success(
        self, tts_user: SimpleNamespace, tmp_path: Path, filesystem_access_enabled: Path
    ) -> None:
        with patch.object(tools_utils, "save_media_to_disk", AsyncMock(side_effect=ToolException("disk exploded"))):
            interface = DeliveringInterface()
            response = await TextToSpeechTool.function(
                text="hello there", save_path=str(tmp_path / "speech.mp3"), interface=interface
            )

        assert response["detail"] == "Audio was successfully sent."
        assert response["saved"] is False
        assert response["save_error"] == "disk exploded"
        interface.send_audio.assert_awaited_once()

    async def test_non_delivering_runner_writes_to_default_media_dir(
        self, tts_user: SimpleNamespace, patched_media_dir: Path, filesystem_access_enabled: Path
    ) -> None:
        interface = StubInterface()
        response = await TextToSpeechTool.function(text="hello there", interface=interface)

        interface.send_audio.assert_not_awaited()
        saved_files = list(patched_media_dir.iterdir())
        assert len(saved_files) == 1
        assert saved_files[0].read_bytes() == b"speech-bytes"
        assert response == {"detail": f"Audio was generated and saved to: {saved_files[0].resolve()}"}

    async def test_non_delivering_runner_with_save_path(
        self, tts_user: SimpleNamespace, tmp_path: Path, filesystem_access_enabled: Path
    ) -> None:
        interface = StubInterface()
        target = tmp_path / "voice.mp3"
        response = await TextToSpeechTool.function(text="hello there", save_path=str(target), interface=interface)

        interface.send_audio.assert_not_awaited()
        assert target.read_bytes() == b"speech-bytes"
        assert response == {"detail": f"Audio was generated and saved to: {target.resolve()}"}

    async def test_delivering_runner_save_with_disabled_filesystem_raises_before_delivery(
        self, tts_user: SimpleNamespace, tmp_path: Path, filesystem_access_disabled: Path
    ) -> None:
        interface = DeliveringInterface()
        with pytest.raises(ToolException, match="FILESYSTEM_ACCESS"):
            await TextToSpeechTool.function(text="hello there", save_path=str(tmp_path / "x.mp3"), interface=interface)

        interface.send_audio.assert_not_awaited()
        assert not (tmp_path / "x.mp3").exists()

    async def test_non_delivering_runner_with_disabled_filesystem_raises(
        self, tts_user: SimpleNamespace, patched_media_dir: Path, filesystem_access_disabled: Path
    ) -> None:
        interface = StubInterface()
        with pytest.raises(PermissionError, match="FILESYSTEM_ACCESS"):
            await TextToSpeechTool.function(text="hello there", interface=interface)

        interface.send_audio.assert_not_awaited()
        assert not patched_media_dir.exists()


class TestImageTool:
    """The image tool implements the response matrix."""

    async def test_delivering_runner_without_save_path_writes_nothing(
        self, image_patches: AsyncMock, patched_media_dir: Path, filesystem_access_enabled: Path
    ) -> None:
        interface = DeliveringInterface()
        response = await GenerateImageTool.function(
            provider="Gemini", image_model="dall-e-3", prompt="a cat", interface=interface
        )

        assert response == {"detail": "Image was successfully generated and sent to user."}
        interface.send_images.assert_awaited_once()
        assert not patched_media_dir.exists()

    async def test_non_delivering_runner_saves_image_bytes(
        self, image_patches: AsyncMock, patched_media_dir: Path, filesystem_access_enabled: Path
    ) -> None:
        interface = StubInterface()
        response = await GenerateImageTool.function(
            provider="Gemini", image_model="dall-e-3", prompt="a cat", interface=interface
        )

        interface.send_images.assert_not_awaited()
        saved_files = list(patched_media_dir.iterdir())
        assert len(saved_files) == 1
        assert saved_files[0].read_bytes() == b"png-bytes"
        assert response == {"detail": f"Image was generated and saved to: {saved_files[0].resolve()}"}


class TestVideoTool:
    """The video tool implements the response matrix."""

    async def test_delivering_runner_save_failure_is_partial_success(
        self, video_patches: AsyncMock, tmp_path: Path, filesystem_access_enabled: Path
    ) -> None:
        with patch.object(tools_utils, "save_media_to_disk", AsyncMock(side_effect=OSError("disk full"))):
            interface = DeliveringInterface()
            response = await GenerateVideoTool.function(prompt="a sunset", save_path=str(tmp_path), interface=interface)

        assert response["detail"] == "Video was successfully generated and sent to user."
        assert response["saved"] is False
        assert response["save_error"] == "disk full"
        interface.send_video.assert_awaited_once()

    async def test_non_delivering_runner_saves_video(
        self, video_patches: AsyncMock, patched_media_dir: Path, filesystem_access_enabled: Path
    ) -> None:
        interface = StubInterface()
        response = await GenerateVideoTool.function(prompt="a sunset", interface=interface)

        interface.send_video.assert_not_awaited()
        saved_files = list(patched_media_dir.iterdir())
        assert len(saved_files) == 1
        assert saved_files[0].suffix == ".mp4"
        assert saved_files[0].read_bytes() == b"mp4-bytes"
        assert response == {"detail": f"Video was generated and saved to: {saved_files[0].resolve()}"}


class TestSunoTools:
    """Both Suno tools implement the response matrix."""

    async def test_delivering_runner_without_save_path_writes_nothing(
        self, suno_provider: SimpleNamespace, patched_media_dir: Path, filesystem_access_enabled: Path
    ) -> None:
        with patch("chibi.services.providers.tools.media.download", AsyncMock(return_value=b"mp3-bytes")):
            interface = DeliveringInterface()
            response = await GenerateMusicViaSunoTool.function(prompt="a song", interface=interface)

        assert response["detail"] == "Music was successfully generated and sent to user"
        assert set(response["suno_task_data"]) == {"task_id", 1, 2}
        interface.send_audio.assert_awaited()
        assert interface.send_audio.await_count == 2
        assert not patched_media_dir.exists()

    async def test_non_delivering_runner_saves_all_versions(
        self, suno_provider: SimpleNamespace, patched_media_dir: Path, filesystem_access_enabled: Path
    ) -> None:
        with patch.object(tools_utils, "download", AsyncMock(return_value=b"mp3-bytes")):
            interface = StubInterface()
            response = await GenerateMusicViaSunoTool.function(prompt="a song", interface=interface)

        interface.send_audio.assert_not_awaited()
        saved_files = sorted(patched_media_dir.iterdir())
        assert len(saved_files) == 2
        assert response["detail"].startswith("Music was generated and saved to: ")
        for saved_file in saved_files:
            assert str(saved_file.resolve()) in response["detail"]

    async def test_custom_mode_non_delivering_runner_saves(
        self, suno_provider: SimpleNamespace, patched_media_dir: Path, filesystem_access_enabled: Path
    ) -> None:
        with patch.object(tools_utils, "download", AsyncMock(return_value=b"mp3-bytes")):
            interface = StubInterface()
            response = await GenerateAdvancedMusicViaSunoTool.function(
                prompt="a song", style="trance", title="Tune", interface=interface
            )

        interface.send_audio.assert_not_awaited()
        assert len(list(patched_media_dir.iterdir())) == 2
        assert response["detail"].startswith("Music was generated and saved to: ")


class TestElevenLabsTool:
    """The ElevenLabs tool implements the response matrix."""

    async def test_delivering_runner_with_save_path(
        self, elevenlabs_provider: SimpleNamespace, tmp_path: Path, filesystem_access_enabled: Path
    ) -> None:
        interface = DeliveringInterface()
        target = tmp_path / "track.mp3"
        response = await GenerateMusicViaElevenLabsTool.function(
            prompt="a melody", title="Track", save_path=str(target), interface=interface
        )

        assert response["detail"] == "Music was successfully generated and sent to user"
        assert response["saved_to"] == [str(target.resolve())]
        assert target.read_bytes() == b"track-bytes"
        interface.send_audio.assert_awaited_once()

    async def test_delivering_runner_with_disabled_filesystem_raises_before_delivery(
        self, elevenlabs_provider: SimpleNamespace, tmp_path: Path, filesystem_access_disabled: Path
    ) -> None:
        interface = DeliveringInterface()
        with pytest.raises(ToolException, match="FILESYSTEM_ACCESS"):
            await GenerateMusicViaElevenLabsTool.function(
                prompt="a melody", title="Track", save_path=str(tmp_path / "t.mp3"), interface=interface
            )

        interface.send_audio.assert_not_awaited()


class TestBackgroundResponseShape:
    """Path-only background responses read as standalone chat messages."""

    async def test_path_only_response_is_suitable_as_chat_message(
        self, tts_user: SimpleNamespace, patched_media_dir: Path, filesystem_access_enabled: Path
    ) -> None:
        interface = StubInterface()
        response = await TextToSpeechTool.function(text="hello there", interface=interface)

        assert isinstance(response["detail"], str)
        assert "Audio was generated and saved to:" in response["detail"]
        assert str(patched_media_dir) in response["detail"]
        assert "saved_to" not in response
