"""Tests for ReadTelegramFileTool."""

from typing import cast
from unittest.mock import AsyncMock, Mock, patch

import pytest

from chibi.exceptions import StorageError
from chibi.services.providers.tools import ReadTelegramFileTool, RegisteredChibiTools


@pytest.fixture
def mock_interface():
    """Create a mock interface."""
    interface = Mock()
    interface.user_id = 12345
    return interface


@pytest.fixture
def mock_kwargs(mock_interface):
    """Create mock kwargs with interface."""
    return {"interface": mock_interface, "user_id": 12345}


class TestReadTelegramFileTool:
    """Tests for ReadTelegramFileTool."""

    def test_tool_registered(self):
        """Test that tool is registered."""
        assert ReadTelegramFileTool.register is True
        assert ReadTelegramFileTool.name == "read_telegram_file"

    def test_tool_present_in_registry(self):
        """Test that the tool is present in the global tool registry after import."""
        assert RegisteredChibiTools.tools_map.get("read_telegram_file") is ReadTelegramFileTool

    def test_definition_has_required_file_id_param(self):
        """Test that tool definition has required file_id parameter."""
        parameters = cast(dict, ReadTelegramFileTool.definition["function"]["parameters"])
        properties = cast(dict, parameters["properties"])
        assert "file_id" in properties
        assert properties["file_id"]["type"] == "string"
        assert parameters["required"] == ["file_id"]

    def test_definition_describes_behavior(self):
        """Test that the LLM-facing description mentions key behaviors."""
        description = ReadTelegramFileTool.definition["function"]["description"]
        assert "file_id" in description
        assert "get_file_info" in description
        assert "truncated" in description
        assert "Non-text" in description

    @patch("chibi.services.providers.tools.telegram_files.get_file_storage")
    async def test_happy_path_returns_content(self, mock_get_storage, mock_kwargs):
        """Test that function returns the content from storage."""
        mock_storage = AsyncMock()
        mock_storage.get_text.return_value = "Hello, this is a text file content."
        mock_get_storage.return_value = mock_storage

        result = await ReadTelegramFileTool.function(file_id="test123", **mock_kwargs)

        mock_get_storage.assert_called_once_with(interface=mock_kwargs["interface"])
        mock_storage.get_text.assert_called_once_with(file_id="test123")
        assert result == {"content": "Hello, this is a text file content."}

    @patch("chibi.services.providers.tools.telegram_files.get_file_storage")
    async def test_truncation_marker_passed_through(self, mock_get_storage, mock_kwargs):
        """Test that truncated content (marker added by storage) is passed through unchanged."""
        mock_storage = AsyncMock()
        mock_storage.get_text.return_value = "a" * 300_000 + "...[truncated 10 of 300010 characters]"
        mock_get_storage.return_value = mock_storage

        result = await ReadTelegramFileTool.function(file_id="test123", **mock_kwargs)

        assert result["content"].endswith("...[truncated 10 of 300010 characters]")

    @patch("chibi.services.providers.tools.telegram_files.get_file_storage")
    async def test_file_not_found_propagates(self, mock_get_storage, mock_kwargs):
        """Test that FileNotFoundError from storage propagates."""
        mock_storage = AsyncMock()
        mock_storage.get_text.side_effect = FileNotFoundError("No file with ID 'missing' found")
        mock_get_storage.return_value = mock_storage

        with pytest.raises(FileNotFoundError, match="missing"):
            await ReadTelegramFileTool.function(file_id="missing", **mock_kwargs)

    @patch("chibi.services.providers.tools.telegram_files.get_file_storage")
    async def test_storage_error_non_text_propagates(self, mock_get_storage, mock_kwargs):
        """Test that StorageError for non-text files propagates."""
        mock_storage = AsyncMock()
        mock_storage.get_text.side_effect = StorageError("File 'photo.bin' is not a text file.")
        mock_get_storage.return_value = mock_storage

        with pytest.raises(StorageError, match="not a text file"):
            await ReadTelegramFileTool.function(file_id="photo.bin", **mock_kwargs)

    @patch("chibi.services.providers.tools.telegram_files.get_file_storage")
    async def test_storage_error_oversize_propagates(self, mock_get_storage, mock_kwargs):
        """Test that StorageError for oversized files propagates."""
        mock_storage = AsyncMock()
        mock_storage.get_text.side_effect = StorageError("File 'big.log' is too large to read as text")
        mock_get_storage.return_value = mock_storage

        with pytest.raises(StorageError, match="too large"):
            await ReadTelegramFileTool.function(file_id="big.log", **mock_kwargs)

    @patch("chibi.services.providers.tools.telegram_files.get_file_storage")
    async def test_value_error_non_telegram_interface_propagates(self, mock_get_storage, mock_kwargs):
        """Test that ValueError for unsupported storage/interface propagates (non-Telegram runners)."""
        mock_get_storage.side_effect = ValueError("Unsupported file storage or user interface type.")

        with pytest.raises(ValueError, match="Unsupported file storage"):
            await ReadTelegramFileTool.function(file_id="test123", **mock_kwargs)
