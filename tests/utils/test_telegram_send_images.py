"""Tests for chibi.utils.telegram.send_images URL, local-path and buffer delivery."""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram import InputFile, InputMediaDocument, InputMediaPhoto
from telegram.constants import FileSizeLimit

from chibi.config import gpt_settings
from chibi.constants import FILE_UPLOAD_TIMEOUT, IMAGE_UPLOAD_TIMEOUT
from chibi.utils.telegram import send_images


def make_update() -> MagicMock:
    """Build a mocked Telegram update for media delivery."""
    update = MagicMock()
    update.update_id = 1
    update.effective_user = MagicMock(id=42, username="tester", name="Tester")
    update.effective_chat = MagicMock(id=555)
    update.effective_message = MagicMock(message_id=7, message_thread_id=None)
    return update


def make_context() -> MagicMock:
    """Build a mocked PTB callback context with an async bot."""
    context = MagicMock()
    context.bot = AsyncMock()
    context.bot.id = 999
    context.user_data = {}
    return context


def make_local_file(tmp_path: Path, size: int = 1024) -> Path:
    """Create a local image file of the given size.

    Args:
        tmp_path: Pytest temporary directory.
        size: File size in bytes.

    Returns:
        The path to the created file.
    """
    file = tmp_path / "picture.png"
    file.write_bytes(b"x" * size)
    return file


class TestSendImagesFromUrls:
    """URL input keeps its existing behaviour byte-for-byte."""

    async def test_urls_downloaded_and_sent_as_media_group(self, monkeypatch: pytest.MonkeyPatch) -> None:
        update = make_update()
        context = make_context()
        monkeypatch.setattr("chibi.utils.telegram.download_image", AsyncMock(return_value=b"img"))

        await send_images(images=["http://example.com/1.png"], update=update, context=context)

        context.bot.send_media_group.assert_awaited_once()
        kwargs: dict[str, Any] = context.bot.send_media_group.await_args.kwargs
        assert kwargs["chat_id"] == 555
        assert kwargs["reply_to_message_id"] == 7
        assert all(isinstance(m, InputMediaPhoto) for m in kwargs["media"])
        context.bot.send_message.assert_not_awaited()


@pytest.fixture
def filesystem_access_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable the filesystem_access gate for the duration of a test."""
    monkeypatch.setattr(gpt_settings, "filesystem_access", True)


class TestSendImagesFromLocalPaths:
    """Local file paths are read from disk and routed through the buffer pipeline."""

    async def test_small_local_file_sent_as_photo(self, tmp_path: Path, filesystem_access_enabled) -> None:
        local_file = make_local_file(tmp_path)
        update = make_update()
        context = make_context()

        await send_images(images=[str(local_file)], update=update, context=context)

        context.bot.send_media_group.assert_awaited_once()
        kwargs: dict[str, Any] = context.bot.send_media_group.await_args.kwargs
        assert kwargs["chat_id"] == 555
        assert kwargs["write_timeout"] == IMAGE_UPLOAD_TIMEOUT
        media = kwargs["media"]
        assert len(media) == 1
        assert isinstance(media[0], InputMediaPhoto)
        media_file = media[0].media
        assert isinstance(media_file, InputFile)
        assert media_file.filename == "picture.png"
        assert media_file.input_file_content == b"x" * 1024

    async def test_large_local_file_sent_as_document(self, tmp_path: Path, filesystem_access_enabled) -> None:
        local_file = make_local_file(tmp_path, size=FileSizeLimit.PHOTOSIZE_UPLOAD + 1)
        update = make_update()
        context = make_context()

        await send_images(images=[str(local_file)], update=update, context=context)

        context.bot.send_media_group.assert_awaited_once()
        kwargs: dict[str, Any] = context.bot.send_media_group.await_args.kwargs
        assert kwargs["write_timeout"] == FILE_UPLOAD_TIMEOUT
        assert all(isinstance(m, InputMediaDocument) for m in kwargs["media"])

    async def test_nonexistent_path_raises_without_delivery(self, tmp_path: Path, filesystem_access_enabled) -> None:
        update = make_update()
        context = make_context()

        with pytest.raises(Exception, match="File not found"):
            await send_images(images=[str(tmp_path / "missing.png")], update=update, context=context)

        context.bot.send_media_group.assert_not_awaited()

    async def test_directory_raises_without_delivery(self, tmp_path: Path, filesystem_access_enabled) -> None:
        update = make_update()
        context = make_context()

        with pytest.raises(Exception, match="File not found"):
            await send_images(images=[str(tmp_path)], update=update, context=context)

        context.bot.send_media_group.assert_not_awaited()

    async def test_filesystem_access_disabled_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gpt_settings, "filesystem_access", False)
        local_file = make_local_file(tmp_path)
        update = make_update()
        context = make_context()

        with pytest.raises(PermissionError, match="Filesystem access is disabled"):
            await send_images(images=[str(local_file)], update=update, context=context)

        context.bot.send_media_group.assert_not_awaited()


class TestSendImagesFailureFallback:
    """The media-group failure fallback never leaks absolute local paths."""

    async def test_fallback_masks_local_paths(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filesystem_access_enabled
    ) -> None:
        local_file = make_local_file(tmp_path)
        update = make_update()
        context = make_context()
        monkeypatch.setattr("chibi.utils.telegram.download_image", AsyncMock(return_value=b"img"))
        context.bot.send_media_group.side_effect = [RuntimeError("upload failed"), None]

        await send_images(images=["http://example.com/1.png", str(local_file)], update=update, context=context)

        context.bot.send_message.assert_awaited_once()
        fallback_text = context.bot.send_message.await_args.kwargs["text"]
        assert "http://example.com/1.png" in fallback_text
        assert "picture.png" in fallback_text
        assert str(tmp_path) not in fallback_text
