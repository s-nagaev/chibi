"""Tests for TelegramFileStorage.get_text and the shared text-file detector."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chibi.exceptions import StorageError
from chibi.models import TelegramFileMeta
from chibi.services.interface import TelegramInterface
from chibi.storage.files.telegram_storage import TelegramFileStorage
from chibi.utils.text_files import is_text_file


def make_file_meta(
    file_name: str = "notes.txt",
    mime_type: str = "text/plain",
    file_size: int = 1024,
) -> TelegramFileMeta:
    """Build a TelegramFileMeta instance with sensible defaults.

    Args:
        file_name: Name of the document.
        mime_type: MIME type reported for the document.
        file_size: Size of the document in bytes.

    Returns:
        The constructed metadata object.
    """
    return TelegramFileMeta(
        file_id="tg-file-1",
        file_name=file_name,
        file_size=file_size,
        mime_type=mime_type,
        file_unique_id="file-unique-1",
    )


def make_storage() -> TelegramFileStorage:
    """Build a TelegramFileStorage backed by mocked Telegram update and context.

    Returns:
        The storage instance with mocked external dependencies.
    """
    update = MagicMock()
    update.effective_user.id = 42
    context = MagicMock()
    return TelegramFileStorage(interface=TelegramInterface(update=update, context=context))


class TestGetText:
    """Tests for TelegramFileStorage.get_text()."""

    async def test_get_text_decodes_utf8(self):
        """Test that a text/plain file is downloaded and decoded as UTF-8."""
        storage = make_storage()
        meta = make_file_meta(file_name="notes.txt", mime_type="text/plain", file_size=5)
        with (
            patch("chibi.storage.files.telegram_storage.get_telegram_document", new=AsyncMock(return_value=meta)),
            patch.object(TelegramFileStorage, "get_bytes", new=AsyncMock(return_value="привет".encode("utf-8"))),
        ):
            result = await storage.get_text("file-unique-1")

        assert result == "привет"

    async def test_get_text_missing_file_raises_file_not_found(self):
        """Test that an unknown file ID raises FileNotFoundError."""
        storage = make_storage()
        with patch("chibi.storage.files.telegram_storage.get_telegram_document", new=AsyncMock(return_value=None)):
            with pytest.raises(FileNotFoundError):
                await storage.get_text("missing-id")

    async def test_get_text_binary_mime_raises_storage_error(self):
        """Test that a non-text file raises StorageError without downloading."""
        storage = make_storage()
        meta = make_file_meta(file_name="archive.bin", mime_type="application/octet-stream")
        get_bytes_mock = AsyncMock(return_value=b"binary")
        with (
            patch("chibi.storage.files.telegram_storage.get_telegram_document", new=AsyncMock(return_value=meta)),
            patch.object(TelegramFileStorage, "get_bytes", new=get_bytes_mock),
        ):
            with pytest.raises(StorageError):
                await storage.get_text("file-unique-1")

        get_bytes_mock.assert_not_awaited()

    async def test_get_text_oversize_raises_storage_error_before_download(self):
        """Test that an oversized text file raises StorageError without calling get_bytes."""
        storage = make_storage()
        meta = make_file_meta(file_name="big.log", mime_type="text/plain", file_size=1_500_000)
        get_bytes_mock = AsyncMock(return_value=b"content")
        with (
            patch("chibi.storage.files.telegram_storage.get_telegram_document", new=AsyncMock(return_value=meta)),
            patch.object(TelegramFileStorage, "get_bytes", new=get_bytes_mock),
        ):
            with pytest.raises(StorageError):
                await storage.get_text("file-unique-1")

        get_bytes_mock.assert_not_awaited()

    async def test_get_text_truncates_with_marker(self):
        """Test that decoded text beyond the character cap is truncated with an explicit marker."""
        storage = make_storage()
        meta = make_file_meta(file_name="notes.txt", mime_type="text/plain")
        long_text = "0123456789ABCDEF"
        with (
            patch("chibi.storage.files.telegram_storage.get_telegram_document", new=AsyncMock(return_value=meta)),
            patch.object(TelegramFileStorage, "get_bytes", new=AsyncMock(return_value=long_text.encode("utf-8"))),
            patch("chibi.storage.files.telegram_storage.MAX_DECODED_TEXT_CHARS", 10),
        ):
            result = await storage.get_text("file-unique-1")

        assert result == "0123456789...[truncated 6 of 16 characters]"

    async def test_get_text_at_cap_is_not_truncated(self):
        """Test that text exactly at the character cap is returned without a marker."""
        storage = make_storage()
        meta = make_file_meta(file_name="notes.txt", mime_type="text/plain")
        exact_text = "0123456789"
        with (
            patch("chibi.storage.files.telegram_storage.get_telegram_document", new=AsyncMock(return_value=meta)),
            patch.object(TelegramFileStorage, "get_bytes", new=AsyncMock(return_value=exact_text.encode("utf-8"))),
            patch("chibi.storage.files.telegram_storage.MAX_DECODED_TEXT_CHARS", 10),
        ):
            result = await storage.get_text("file-unique-1")

        assert result == "0123456789"


class TestIsTextFile:
    """Tests for the shared is_text_file() helper."""

    def test_text_mime_is_positive(self):
        """Test that a text/* MIME type is detected as text regardless of extension."""
        assert is_text_file(mime_type="text/plain", file_name="archive.bin") is True

    def test_allowlisted_mime_is_positive(self):
        """Test that an allowlisted application MIME type is detected as text."""
        assert is_text_file(mime_type="application/json", file_name=None) is True

    def test_extension_fallback_is_positive(self):
        """Test that a known text extension is detected via the fallback path."""
        assert is_text_file(mime_type="application/octet-stream", file_name="README.md") is True

    def test_extension_without_mime_is_positive(self):
        """Test that a known text extension is detected when MIME type is None."""
        assert is_text_file(mime_type=None, file_name="script.py") is True

    def test_binary_mime_with_unknown_extension_is_negative(self):
        """Test that a binary MIME type with an unknown extension is not text."""
        assert is_text_file(mime_type="image/png", file_name="photo.xyz") is False

    def test_unknown_extension_is_negative(self):
        """Test that an unknown extension with no MIME type is not text."""
        assert is_text_file(mime_type=None, file_name="data.xyz") is False

    def test_both_none_is_negative(self):
        """Test that missing MIME type and file name is not text."""
        assert is_text_file(mime_type=None, file_name=None) is False
