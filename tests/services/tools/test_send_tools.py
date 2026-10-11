"""Tests for send_image, send_video and send_audio tools with URL or local path input."""

from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from chibi.config import gpt_settings
from chibi.services.providers.tools.exceptions import ToolException
from chibi.services.providers.tools.send import SendAudioTool, SendImageTool, SendVideoTool
from chibi.services.providers.tools.tool import ChibiTool
from chibi.services.providers.tools.utils import is_remote_url, resolve_local_file


def parameter_description(tool: type[ChibiTool], parameter: str) -> str:
    """Extract a schema description for a tool parameter.

    Args:
        tool: The tool class to inspect.
        parameter: The parameter name in the tool schema.

    Returns:
        The description string of the parameter.
    """
    definition = cast(dict[str, Any], tool.definition["function"])
    properties = cast(dict[str, Any], definition["parameters"]["properties"])
    return str(properties[parameter]["description"])


@pytest.fixture
def interface() -> AsyncMock:
    """A mock user interface capturing media delivery calls."""
    return AsyncMock()


@pytest.fixture
def mock_kwargs(interface: AsyncMock) -> dict[str, Any]:
    """Invocation kwargs carrying the mock interface."""
    return {"interface": interface}


@pytest.fixture
def local_file(tmp_path: Path) -> Path:
    """An existing local media file."""
    file = tmp_path / "media.png"
    file.write_bytes(b"media-bytes")
    return file


class TestIsRemoteUrl:
    """The shared URL-or-path detector."""

    def test_http_url_detected(self) -> None:
        assert is_remote_url("http://example.com/a.png") is True

    def test_https_url_detected(self) -> None:
        assert is_remote_url("https://example.com/a.png") is True

    def test_absolute_path_not_a_url(self) -> None:
        assert is_remote_url("/tmp/a.png") is False

    def test_tilde_path_not_a_url(self) -> None:
        assert is_remote_url("~/a.png") is False

    def test_relative_path_not_a_url(self) -> None:
        assert is_remote_url("a.png") is False


class TestResolveLocalFile:
    """The shared local-path resolver with the filesystem_access gate."""

    def test_resolves_existing_file(self, local_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gpt_settings, "filesystem_access", True)
        assert resolve_local_file(str(local_file)) == local_file.resolve()

    def test_expands_tilde(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gpt_settings, "filesystem_access", True)
        home_dir = tmp_path / "home"
        home_dir.mkdir()
        target = home_dir / "a.png"
        target.write_bytes(b"x")
        monkeypatch.setenv("HOME", str(home_dir))

        assert resolve_local_file("~/a.png") == target.resolve()

    def test_nonexistent_path_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gpt_settings, "filesystem_access", True)

        with pytest.raises(ToolException, match="File not found"):
            resolve_local_file(str(tmp_path / "missing.png"))

    def test_directory_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gpt_settings, "filesystem_access", True)

        with pytest.raises(ToolException, match="File not found"):
            resolve_local_file(str(tmp_path))

    def test_requires_filesystem_access(self, local_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gpt_settings, "filesystem_access", False)

        with pytest.raises(PermissionError, match="Filesystem access is disabled"):
            resolve_local_file(str(local_file))


class TestSendImageTool:
    """send_image accepts an http(s) URL or a local file path."""

    def test_definition_keeps_parameter_name(self) -> None:
        assert "local file" in parameter_description(tool=SendImageTool, parameter="image_url")

    async def test_url_passthrough_unchanged(self, interface: AsyncMock, mock_kwargs: dict[str, Any]) -> None:
        result = await SendImageTool.function(image_url="http://example.com/a.png", **mock_kwargs)

        assert result == {"detail": "Image was successfully sent."}
        interface.send_images.assert_awaited_once_with(images=["http://example.com/a.png"], reply=False, caption=None)

    async def test_local_path_delivers_file(
        self, interface: AsyncMock, mock_kwargs: dict[str, Any], local_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gpt_settings, "filesystem_access", True)

        await SendImageTool.function(image_url=str(local_file), **mock_kwargs)

        interface.send_images.assert_awaited_once_with(images=[str(local_file.resolve())], reply=False, caption=None)

    async def test_nonexistent_path_rejected(
        self, interface: AsyncMock, mock_kwargs: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gpt_settings, "filesystem_access", True)

        with pytest.raises(ToolException, match="File not found"):
            await SendImageTool.function(image_url=str(tmp_path / "missing.png"), **mock_kwargs)

        interface.send_images.assert_not_awaited()

    async def test_directory_rejected(
        self, interface: AsyncMock, mock_kwargs: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gpt_settings, "filesystem_access", True)

        with pytest.raises(ToolException, match="File not found"):
            await SendImageTool.function(image_url=str(tmp_path), **mock_kwargs)

        interface.send_images.assert_not_awaited()

    async def test_filesystem_access_disabled(
        self, interface: AsyncMock, mock_kwargs: dict[str, Any], local_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gpt_settings, "filesystem_access", False)

        with pytest.raises(PermissionError, match="Filesystem access is disabled"):
            await SendImageTool.function(image_url=str(local_file), **mock_kwargs)

        interface.send_images.assert_not_awaited()


class TestSendVideoTool:
    """send_video accepts an http(s) URL or a local file path."""

    def test_definition_keeps_parameter_name(self) -> None:
        assert "local file" in parameter_description(tool=SendVideoTool, parameter="video_url")

    async def test_url_passthrough_unchanged(self, interface: AsyncMock, mock_kwargs: dict[str, Any]) -> None:
        await SendVideoTool.function(video_url="http://example.com/a.mp4", **mock_kwargs)

        interface.send_video.assert_awaited_once_with(
            video="http://example.com/a.mp4", caption=None, duration=None, thumbnail=None
        )

    async def test_local_path_passed_to_transport(
        self, interface: AsyncMock, mock_kwargs: dict[str, Any], local_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gpt_settings, "filesystem_access", True)

        await SendVideoTool.function(video_url=str(local_file), **mock_kwargs)

        interface.send_video.assert_awaited_once_with(
            video=str(local_file.resolve()), caption=None, duration=None, thumbnail=None
        )

    async def test_nonexistent_path_rejected(
        self, interface: AsyncMock, mock_kwargs: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gpt_settings, "filesystem_access", True)

        with pytest.raises(ToolException, match="File not found"):
            await SendVideoTool.function(video_url=str(tmp_path / "missing.mp4"), **mock_kwargs)

        interface.send_video.assert_not_awaited()

    async def test_filesystem_access_disabled(
        self, interface: AsyncMock, mock_kwargs: dict[str, Any], local_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gpt_settings, "filesystem_access", False)

        with pytest.raises(PermissionError, match="Filesystem access is disabled"):
            await SendVideoTool.function(video_url=str(local_file), **mock_kwargs)

        interface.send_video.assert_not_awaited()


class TestSendAudioTool:
    """send_audio accepts an http(s) URL or a local file path."""

    def test_definition_keeps_parameter_name(self) -> None:
        assert "local file" in parameter_description(tool=SendAudioTool, parameter="audio_url")

    async def test_url_downloaded_as_before(
        self, interface: AsyncMock, mock_kwargs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_download(url: str) -> bytes | None:
            return b"audio-bytes"

        monkeypatch.setattr("chibi.services.providers.tools.send.download", fake_download)

        await SendAudioTool.function(audio_url="http://example.com/a.mp3", title="Track", **mock_kwargs)

        interface.send_audio.assert_awaited_once()
        assert interface.send_audio.await_args.kwargs["audio"] == b"audio-bytes"

    async def test_url_download_failure_keeps_url_behaviour(
        self, interface: AsyncMock, mock_kwargs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def failing_download(url: str) -> bytes | None:
            return None

        monkeypatch.setattr("chibi.services.providers.tools.send.download", failing_download)

        await SendAudioTool.function(audio_url="http://example.com/a.mp3", **mock_kwargs)

        assert interface.send_audio.await_args.kwargs["audio"] == "http://example.com/a.mp3"

    async def test_local_path_passed_to_transport(
        self, interface: AsyncMock, mock_kwargs: dict[str, Any], local_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gpt_settings, "filesystem_access", True)

        await SendAudioTool.function(audio_url=str(local_file), **mock_kwargs)

        interface.send_audio.assert_awaited_once()
        assert interface.send_audio.await_args.kwargs["audio"] == str(local_file.resolve())

    async def test_nonexistent_path_rejected(
        self, interface: AsyncMock, mock_kwargs: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gpt_settings, "filesystem_access", True)

        with pytest.raises(ToolException, match="File not found"):
            await SendAudioTool.function(audio_url=str(tmp_path / "missing.mp3"), **mock_kwargs)

        interface.send_audio.assert_not_awaited()

    async def test_filesystem_access_disabled(
        self, interface: AsyncMock, mock_kwargs: dict[str, Any], local_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gpt_settings, "filesystem_access", False)

        with pytest.raises(PermissionError, match="Filesystem access is disabled"):
            await SendAudioTool.function(audio_url=str(local_file), **mock_kwargs)

        interface.send_audio.assert_not_awaited()
